# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys
import math
import os
from turtle import down
from typing import Iterable
import imageio
import numpy as np

import torch

import util.misc as misc
import util.lr_sched as lr_sched

from util.mri_tools import rifft2
from util.metric import calc_metrics


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train=True
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        """ if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        """
        samples = data['down'].to(device, non_blocking=True)
        ssl_masks = data['mask'].to(device, non_blocking=True)
        full_samples = data['full'].to(device, non_blocking=True)

        if args.autocast:
            with torch.cuda.amp.autocast():
                sploss, sslloss, pred, mask = model(samples, ssl_masks, full_samples, mask_ratio=args.mask_ratio)
        else: 
            sploss, sslloss, pred, mask = model(samples, ssl_masks, full_samples, mask_ratio=args.mask_ratio)

        # spatial domain loss
        if args.downsample>1:
            pred_dc = samples + pred*ssl_masks
        else:
            mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1,2,1,256)
            pred_dc = samples*(1-mask) + pred*mask # 0 is keep, 1 is remove

        pred_dc, full = rifft2(pred_dc[:,:,:,:], full_samples[:,:,:,:], permute=True) 
        maxnum = torch.max(full)
        minnum = torch.min(full)
        pred_dc = (pred_dc-minnum)/(maxnum-minnum+1e-08)
        full = (full-minnum)/(maxnum-minnum+1e-08)
        imgloss = torch.sum(torch.abs(pred_dc-full))/samples.shape[0]/256      
        # imgloss = torch.tensor([0], device=sploss.device)

        loss = sploss + args.ssl_weight*sslloss + args.img_weight*imgloss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),     #clip_grad=1
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(sploss=sploss.item())
        metric_logger.update(sslloss=sslloss.item())
        metric_logger.update(imgloss=imgloss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        sploss_value_reduce = misc.all_reduce_mean(sploss.item())
        sslloss_value_reduce = misc.all_reduce_mean(sslloss.item())
        imgloss_value_reduce = misc.all_reduce_mean(imgloss.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_sploss', sploss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_sslloss', sslloss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_imgloss', imgloss_value_reduce, epoch_1000x)
            

            log_writer.add_scalar('lr', lr, epoch_1000x)
        # i=0
        # for name, param in model.named_parameters():
        #     i+=1
        #     if i==9 and param.requires_grad:
        #         print(name)
        #         print('during model.params: \n', param.data)
        #         print('during model.params.grad: \n', param.grad)
        #         break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, 
                    epoch: int,
                    args=None):

    model.train=False
    keys = ['noise_loss', 'loss', 'noise_psnr', 'psnr', 'psnr_dc', 'noise_ssim', 'ssim', 'ssim_dc', 'noise_nmse', 'nmse']
    valid_stats = {k:0 for k in keys}
    vnum = len(data_loader)

    save_folder = os.path.join(args.output_dir, 'valid_epoch{:02d}'.format(epoch))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    """ 
    num_low_freqs = 44
    num_high_freqs = 20
    h=256
    center_mask = np.zeros(256, dtype=np.float32)
    pad = (256 - num_low_freqs + 1) // 2
    center_mask[pad : pad+num_low_freqs]=1
    assert center_mask.sum() == num_low_freqs
    center_mask = torch.tensor(center_mask).view(1,-1,1)
    
    adjusted_accel = int((h-num_low_freqs)/(num_high_freqs))
    accel_mask = np.zeros(h, dtype=np.float32)
    accel_mask[0::adjusted_accel]=1
    accel_mask = torch.tensor(accel_mask).view(1,-1,1)
    mask = torch.max(center_mask, accel_mask) # 1 is keep, 0 is remove
    smasks = torch.ones(2,256,256)*(1-mask)
    """

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full_samples = data['full'].to(device, non_blocking=True)

            """ if args.downsample < 2:
                samples = samples*(mask.to(samples.device))
                ssl_masks = smasks.to(samples.device) """

            _, _, pred, _  = model(samples, ssl_masks, full_samples, mask_ratio=0)

            # Data consistency
            pred = torch.clamp(pred, min=-1, max=1)
            pred_dc = samples + pred*ssl_masks
            
            samples, pred, pred_dc, full = rifft2(samples[0,:,:,:], pred[0,:,:,:], pred_dc[0,:,:,:], full_samples[0,:,:,:], permute=True) 
            
            #normalization [0-1]
            max = torch.max(samples)
            min = torch.min(samples)
            samples = torch.clamp((samples-min)/(max-min), min=0, max=1)
            pred = torch.clamp((pred-min)/(max-min), min=0, max=1)
            pred_dc = torch.clamp((pred_dc-min)/(max-min), min=0, max=1)
            full = torch.clamp((full-min)/(max-min), min=0, max=1)
            
            #calculate psnr, ssim
            stat = calc_metrics(samples.unsqueeze(0), pred.unsqueeze(0), full.unsqueeze(0))
            stat_dc = calc_metrics(samples.unsqueeze(0), pred_dc.unsqueeze(0), full.unsqueeze(0))
            for k,v in stat.items():
                valid_stats[k]+=v/vnum
            valid_stats['psnr_dc'] += stat_dc['psnr']/vnum
            valid_stats['ssim_dc'] += stat_dc['ssim']/vnum
            
            #image save
            if i%100==0:
                imageio.imwrite(os.path.join(save_folder, 'ep{:02d}_concat_{:03d}.tif'.format(epoch, int(i/100))), torch.cat([samples, pred, pred_dc, full], dim=-1).squeeze().cpu().numpy())

    print('Validation Epoch: {} {}'.format(epoch, ', '.join(['{}: {:.3f}'.format(k,v.item()) for k,v in valid_stats.items()])))
    return valid_stats

