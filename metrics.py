# -*- coding: utf-8 -*-
# This is code based on https://sudomake.ai/inception-score-explained/.
# as well as https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/metrics.py
import os, time, argparse, math, random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, ImageNet, QMNIST
from torchvision import transforms
from torch.nn import functional as F
from torch.autograd import Variable

import lpips

# some metrics implementation for model inversion:
# PSNR, LPIPS, SSIM

def PSNR(img_batch, ref_batch, max_I=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        # get psnr for a pair of single img_in and img_ref
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(max_I**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    img_is_batched = True if len(img_batch.shape) == 4 else False
    ref_is_batched = True if len(ref_batch.shape) == 4 else False

    if img_is_batched and ref_is_batched:
        assert img_batch.shape == ref_batch.shape

        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()
    elif img_is_batched:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch))
        psnr = torch.stack(psnrs, dim=0).mean()
    elif ref_is_batched:
        [B, C, m, n] = ref_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach(), ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()
    else:
        psnr = get_psnr(img_batch.detach(), ref_batch)

    '''
    if len(img_batch.shape) == 4: # multi-image PSNR
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()
    else: # single-image PSNR
        psnr = get_psnr(img_batch.detach(), ref_batch)
    '''

    return psnr.item()

def lpips_loss(img_batch, ref_batch, net='alex'):
    loss_fn = lpips.LPIPS(net=net)
    [B, C, m, n] = img_batch.shape
    lpips_losses = []
    for sample in range(B):
        lpips_losses.append(loss_fn(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
    lpips_loss = torch.stack(lpips_losses, dim=0).mean()

    return lpips_loss.item()

def ssim_ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def ssim_create_window(window_size, channel):
    _1D_window = ssim_gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = ssim_create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return ssim_ssim(img1, img2, window, window_size, channel, size_average)

def ssim_batch(ref_batch, img_batch, batched=False, factor=1.0):

    [B, C, m, n] = img_batch.shape
    ssims = []
    for sample in range(B):
        ssims.append(ssim(img_batch.detach()[sample, :, :, :].unsqueeze(0), ref_batch[sample, :, :, :].unsqueeze(0)))
    
    mean_ssim = torch.stack(ssims, dim=0).mean()
    return mean_ssim.item(), ssims

def tv_loss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss
