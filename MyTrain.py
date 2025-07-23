import torch
import argparse
import torch.nn.functional as F
from Src.model.SINet.SINet import SINet_ResNet50
from Src.model.SINetV2.Network_Res2Net_GRA_NCD import Network
from Src.utils.Dataloader import get_srcloader, get_tarloader, test_dataset
from Src.utils.tool import structure_loss, clip_gradient, set_random_seed, adjust_lr, update_ema
from torch.autograd import Variable
from datetime import datetime
from Src.utils.tool import ESLoss
from CLS import cls
import numpy as np
import random
import os


def trainer(source_loader, target_loader, model, ema_model, optimizer, epoch, opt, loss_func, total_step, alpha, log_file_path):
    global global_step
    model.train()
    ema_model.train()

    cuda = torch.device('cuda')
    for step, ((src_image, src_gt), (tar_weak_image, tar_strong_image)) in enumerate(zip(source_loader, target_loader)):
        optimizer.zero_grad()
        src_image, tar_weak_image, tar_strong_image = Variable(src_image).cuda(), Variable(tar_weak_image).cuda(),Variable(tar_strong_image).cuda()
        src_gt = Variable(src_gt).cuda()

        if opt.backbone == 'SINet':
            cam_sm, cam_im= model(src_image)

            with torch.no_grad():
                _, tea_out= ema_model(tar_weak_image)
            tea_out = tea_out.sigmoid()

            _, stu_out= model(tar_strong_image)
            stu_out  = stu_out.sigmoid()

            # Supervised loss on source, consistency loss on target
            loss_sup = loss_func(cam_sm, src_gt) + loss_func(cam_im, src_gt)
            loss_con = ES_Loss(stu_out, tea_out)
        
        elif opt.backbone == 'SINet-v2':
            preds = model(src_image)
            with torch.no_grad():
                _, _, _, tea_out = ema_model(tar_weak_image)
            tea_out = tea_out.sigmoid()

            _, _, _, stu_out = model(tar_strong_image)
            stu_out = stu_out.sigmoid()

            loss_sup = structure_loss(preds[0], src_gt) + structure_loss(preds[1], src_gt) + structure_loss(preds[2], src_gt) + structure_loss(preds[3], src_gt)
            loss_con = ES_Loss(stu_out, tea_out)
        
        else:
            raise ValueError(f"Unsupported model type: {opt.backbone}")

        loss_total = loss_sup + loss_con

        loss_total.backward()
        if clip_grad:
            clip_gradient(optimizer, opt.clip)
        optimizer.step()
        global_step += 1
        update_ema(model, ema_model, alpha)

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_all: {:.4f} Loss_sup: {:0.4f} Loss_con: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_total.data, loss_sup.data, loss_con.data))
            log_message = '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_all: {:.4f} Loss_sup: {:0.4f} Loss_con: {:0.4f}]\n'.format(
                datetime.now(), epoch, opt.epoch, step, total_step, loss_total.data, loss_sup.data, loss_con.data)
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_message)
                log_file.flush()


    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)


    if (epoch+1) % opt.save_epoch == 0:
        torch.save(ema_model.state_dict(), save_path + 'Tea_%d.pth' % (epoch+1))
        torch.save(model.state_dict(), save_path + 'Stu_%d.pth' % (epoch + 1))

    if (epoch+1) > 35 and opt.backbone == 'SINet':
        torch.save(ema_model.state_dict(), save_path + 'Tea_%d.pth' % (epoch+1))
        torch.save(model.state_dict(), save_path + 'Stu_%d.pth' % (epoch + 1))

def val(test_loader, ema_model, backbone, epoch, save_path):
    """
    validation function
    """
    global best_teamae, best_epoch
    ema_model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            if backbone == 'SINet':
                _, res = ema_model(image)
            elif backbone == 'SINet-v2':
                res_all = ema_model(image)
                res = res_all[3]

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {}, MAE: {}, bestteaMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_teamae, best_epoch))
        if epoch == 1:
            best_teamae = mae
        else:
            if mae < best_teamae:
                best_teamae = mae
                best_epoch = epoch
                torch.save(ema_model.state_dict(), save_path + 'Tea_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='SINet', choices=['SINet', 'SINet-v2'], help='Select the backbone model architecture.')
    parser.add_argument('--epoch', type=int, default=40, help='epoch number, default=40')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352, help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10, help='every N epochs save your trained snapshot')
    parser.add_argument('--iteration', type=int, default=2)
    parser.add_argument('--task', type=str, default='S2C',choices=['C2C', 'S2C'], 
                        help="Select the task type: 'C2C' or 'S2C'. ""This determines the source dataset and associated hyperparameters.")
    parser.add_argument('--alpha', type=float, default=0.9996, help='EMA update momentum: teacher-student model smoothing factor')
    parser.add_argument('--u', type=float, default=1.0, help='cls module hyperparameter u ')
    parser.add_argument('--tau', type=float, default=0.5, help='cls module hyperparameter tau (threshold)')
    parser.add_argument('--a', type=float, default=0.7, help='EdgeAwareLoss hyperparameter a')
    parser.add_argument('--b', type=float, default=0.3, help='EdgeAwareLoss hyperparameter b')
    parser.add_argument('--c', type=float, default=0.5, help='EdgeAwareLoss hyperparameter c')
    parser.add_argument('--save_model', type=str, default='./Snapshot/SINet/test/')
    parser.add_argument('--source_root', type=str, default='./Dataset/Source/COD10K/')
    parser.add_argument('--target_root', type=str, default='./Dataset/Target/')
    parser.add_argument('--val_root', type=str, default='./Dataset/Val/CAMO/', help='the test rgb images root')
    opt = parser.parse_args()

    # Override hyperparameters for S2C
    if opt.task == 'S2C':
        opt.alpha = 0.996
        opt.u = 0.8
        opt.tau = 0.4
        opt.a = 0.9
        opt.b = 0.3
        opt.c = 0.5
        opt.source_root = './Dataset/Source/HKU-IS/'

    torch.cuda.set_device(opt.gpu)

    set_random_seed(42)  

    # Cycling
    for i in range(1, opt.iteration + 1):
        print(f"\n{'='*20} Iteration {i}/{opt.iteration} started {'='*20}\n")

        # Initialize student model and teacher model
        if opt.backbone == 'SINet':
            model = SINet_ResNet50(channel=32).cuda()
            model_ema = SINet_ResNet50(channel=32).cuda()
            clip_grad = False
        elif opt.backbone == 'SINet-v2':
            model = Network(channel=32).cuda()
            model_ema = Network(channel=32).cuda()
            opt.epoch = 100
            opt.batchsize = 32
            opt.decay_epoch = 50
            clip_grad = True
        else:
            raise ValueError(f"Unsupported model type: {opt.backbone}")
        print(f'[Info] Using backbone: {opt.backbone}')

        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.require_grad = False


        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

        LogitsBCE = torch.nn.BCEWithLogitsLoss()
        ES_Loss = ESLoss(a=opt.a, b=opt.b, c=opt.c).cuda()
        PGT_Loss = ESLoss(a=opt.a, b=opt.b, c=opt.c, use_weighted_bce=False).cuda()

        global_step = 0
        best_teamae = 1
        best_epoch = 0

        source_loader = get_srcloader(image_root=opt.source_root + 'Image/',
                                gt_root=opt.source_root + 'GT/',
                                batchsize=opt.batchsize,
                                trainsize=opt.trainsize,
                                num_workers=6)
        target_loader = get_tarloader(image_root=opt.target_root + 'Image/',
                                    batchsize=opt.batchsize,
                                    trainsize=opt.trainsize,
                                    num_workers=6)

        val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                                gt_root=opt.val_root + 'GT/',
                                testsize=opt.trainsize)
        total_step = min(len(source_loader), len(target_loader))

        log_file_path = os.path.join(opt.save_model, 'training_log.log')
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        with open(log_file_path, 'a') as log_file:
            log_file.write('Training Log\n')
        
        # train and eval
        for epoch_iter in range(1, opt.epoch):
            adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)

            trainer(source_loader=source_loader, target_loader=target_loader, 
                    model=model, ema_model=model_ema, optimizer=optimizer, epoch=epoch_iter,
                    opt=opt, loss_func=LogitsBCE, total_step=total_step, alpha=opt.alpha, log_file_path=log_file_path)
            if epoch_iter > 20:
                val_loader.index = 0
                val(test_loader=val_loader, ema_model=model_ema, backbone=opt.backbone, epoch=epoch_iter, save_path=opt.save_model)

        # Confident label selection
        if i < opt.iteration:
            new_source_root = cls(opt.save_model, opt.source_root, opt.source_root, opt.target_root, PGT_Loss, opt.u, opt.tau, iteration=i, backbone=opt.backbone)
            opt.source_root = new_source_root