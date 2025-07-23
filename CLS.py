import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
import shutil
from PIL import Image
from Src.model.SINet.SINet import SINet_ResNet50
from Src.model.SINetV2.Network_Res2Net_GRA_NCD import Network
from Src.utils.Dataloader import test_dataset

def cls(model_path, source_root, gt_root, target_root, ES_loss, u, tau, iteration=1, testsize=352, dataset_name='COD10K', backbone='SINet'):

    # Copy source and GT directories to avoid overwriting
    source_copy_root = source_root.rstrip('/\\') + f'_iteration{iteration + 1}/'
    gt_copy_root = gt_root.rstrip('/\\') + f'_iteration{iteration + 1}/'

    if os.path.exists(source_copy_root):
        print(f'[Info] Source folder already exists. Removing: {source_copy_root}')
        shutil.rmtree(source_copy_root)
    print(f'[Info] Copying source image folder: {source_root} -> {source_copy_root}')
    shutil.copytree(source_root, source_copy_root)

    if os.path.exists(gt_copy_root):
        print(f'[Info] GT folder already exists. Removing: {gt_copy_root}')
        shutil.rmtree(gt_copy_root)
    print(f'[Info] Copying GT folder: {gt_root} -> {gt_copy_root}')
    shutil.copytree(gt_root, gt_copy_root)

    # Define save paths for images and pseudo-labels
    image_save_dir = os.path.join(source_copy_root, 'Image')
    pgt_save_dir = os.path.join(gt_copy_root, 'GT')

    # Ensure directories exist
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(pgt_save_dir, exist_ok=True)

    # Load student and teacher models
    if backbone == 'SINet':
        model = SINet_ResNet50().cuda()
        ema_model = SINet_ResNet50().cuda()
        stu_ckpt_name = 'Stu_40.pth'
        tea_fallback_path = os.path.join(model_path, 'Tea_40.pth')
    elif backbone == 'SINet-v2':
        model = Network().cuda()
        ema_model = Network().cuda()
        stu_ckpt_name = 'Stu_100.pth'
        tea_fallback_path = os.path.join(model_path, 'Tea_100.pth')
    else:
        raise NotImplementedError(f'Backbone {backbone} not supported.')
    

    model.load_state_dict(torch.load(os.path.join(model_path, stu_ckpt_name)))

    tea_best_path = os.path.join(model_path, 'Tea_epoch_best.pth')


    if os.path.exists(tea_best_path):
        print(f'[Info] Loading EMA model from: {tea_best_path}')
        ema_model.load_state_dict(torch.load(tea_best_path))
    elif os.path.exists(tea_fallback_path):
        print(f'[Warning] TeaNet_epoch_best.pth not found, using fallback: {tea_fallback_path}')
        ema_model.load_state_dict(torch.load(tea_fallback_path))
    else:
        raise FileNotFoundError('[Error] Neither TeaNet_epoch_best.pth nor TeaSINet_40.pth was found.')
    
    model.eval()
    ema_model.eval()

    # First pass: compute average edge loss
    test_loader = test_dataset(image_root=target_root + 'Image/',
                               gt_root=None,
                               testsize=testsize,
                               mode='cls')
    avg_loss = 0.0
    for idx in range(test_loader.size):
        image, name, original_image = test_loader.load_data()
        image = image.cuda()

        if backbone == 'SINet':
            _, stu = model(image)
            _, tea = ema_model(image)
        elif backbone == 'SINet-v2':
            stu_all = model(image)
            tea_all = ema_model(image)
            stu = stu_all[3]
            tea = tea_all[3]
        
        stu1 = stu.sigmoid()
        tea1 = tea.sigmoid()

        edge_loss = ES_loss(stu1, tea1)
        avg_loss += edge_loss.item()

        if (idx + 1) % 10 == 0 or (idx + 1) == test_loader.size:
            print(f"[Progress] Pass 1: {idx + 1}/{test_loader.size} images processed.")

    avg_loss /= test_loader.size
    print('[Info] Average edge loss: {:.6f}'.format(avg_loss))

    # Second pass: generate and save pseudo-labels
    test_loader = test_dataset(image_root=target_root + 'Image/',
                               gt_root=None,
                               testsize=testsize,
                               mode='cls')
    img_count = 1
    for _ in range(test_loader.size):
        image,  name, original_image = test_loader.load_data()
        image = image.cuda()

        if backbone == 'SINet':
            _, stu = model(image)
            _, tea = ema_model(image)
        elif backbone == 'SINet-v2':
            stu_all = model(image)
            tea_all = ema_model(image)
            stu = stu_all[3]
            tea = tea_all[3]
        stu1 = stu.sigmoid()
        tea1 = tea.sigmoid()

        edge_loss = ES_loss(stu1, tea1)
        if edge_loss.item() < u * avg_loss:
            if backbone == 'SINet':
                _, cam = ema_model(image)
            elif backbone == 'SINet-v2':
                _, _, _, cam = ema_model(image)
            cam = F.interpolate(cam, size=(original_image.size[1], original_image.size[0]), mode='bilinear', align_corners=True)
            cam = cam.sigmoid().data.cpu().numpy().squeeze()

            if np.max(cam) < tau:
                print(f'[Skip] CAM: {name}')
                continue
            cam[cam < tau] = 0
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            cv2.imwrite(os.path.join(pgt_save_dir, name), cam * 255)
            original_image.save(os.path.join(image_save_dir, name))

            print(f'[PGT] Dataset: {dataset_name}, Image: {name} ({img_count}/{test_loader.size})')
            img_count += 1

    print("\n[✓] PGT generation completed.")
    return source_copy_root  # Return updated source 