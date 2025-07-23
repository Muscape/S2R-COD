import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()

def numpy2tensor(numpy):
    return torch.from_numpy(numpy).cuda()

def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def update_ema(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

class ESLoss(nn.Module):
    def __init__(self, a=0.7, b=0.3, c=0.5, use_weighted_bce=True):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.use_weighted_bce = use_weighted_bce
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

    def forward(self, pred, target):
        # Extract edges
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        edge_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)

        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)
        edge_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)

        # Edge-Aware loss
        edge_loss = F.l1_loss(edge_pred, edge_target)
        
        # Saliency-Weighted loss
        if self.use_weighted_bce:
            saliency_weight = target + self.c
            region_loss = F.binary_cross_entropy(pred, target, weight=saliency_weight)
        else:
            region_loss = F.binary_cross_entropy(pred, target)
        
        return self.a * edge_loss + self.b * region_loss