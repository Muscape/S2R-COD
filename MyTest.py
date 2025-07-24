import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from Src.model.SINet.SINet import SINet_ResNet50
from Src.model.SINetV2.Network_Res2Net_GRA_NCD import Network
from Src.utils.Dataloader import test_dataset
from Src.utils.tool import eval_mae, numpy2tensor
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='SINet-v2', choices=['SINet', 'SINet-v2'], help='Select the model architecture.')
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/SINet-v2/test/Tea_epoch_best.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/SINet-v2/test/')
opt = parser.parse_args()


if opt.network == 'SINet':
    model = SINet_ResNet50().cuda()
elif opt.network == 'SINet-v2':
    model = Network().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()


for dataset in ['COD10K']:
    save_path = opt.test_save + '/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset(image_root='./Dataset/Test/Image/'.format(dataset),
                               gt_root='./Dataset/Test/GT/'.format(dataset),
                               testsize=opt.testsize,
                               mode='test')
    img_count = 1
    avg_mae = 0.0
    for iteration in range(test_loader.size):
        # load data
        image,  gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        if opt.network == 'SINet':
            _, cam = model(image)
        elif opt.network == 'SINet-v2':
            _, _, _, res2 = model(image)
            cam = res2
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cv2.imwrite(save_path+name, cam*255)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        avg_mae += mae
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

avg_mae /= test_loader.size
print("\n[Congratulations! Testing Done]")
print("\nAvg_MAE: {}".format(avg_mae))