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
                    log_writer=None, lr_scheduler=None,
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

        if args.lr_scheduler=='base':
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
       
        samples = data['down'].to(device, non_blocking=True)
        ssl_masks = data['mask'].to(device, non_blocking=True)
        full_samples = data['full'].to(device, non_blocking=True)

        if args.autocast:
            with torch.cuda.amp.autocast():
                sploss, imgloss, sslloss, advloss = model(down=samples, ssl_masks=ssl_masks, full=full_samples, mask_ratio=args.mask_ratio)
        else: 
            sploss, imgloss, sslloss, advloss = model(down=samples, ssl_masks=ssl_masks, full=full_samples, mask_ratio=args.mask_ratio)

        loss = sploss + args.ssl_weight*sslloss + args.img_weight*imgloss + args.adv_weight*advloss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),     #clip_grad=1
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # ema update
        if 'ema' in args.model:
            for i in range(len(model.ema_blocks)):
                model.ema_blocks[i].update()
                
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(sploss=sploss.item())
        metric_logger.update(imgloss=imgloss.item())
        metric_logger.update(sslloss=sslloss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        # sploss_value_reduce = misc.all_reduce_mean(sploss.item())
        # sslloss_value_reduce = misc.all_reduce_mean(sslloss.item())
        # imgloss_value_reduce = misc.all_reduce_mean(imgloss.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            sploss_value_reduce = misc.all_reduce_mean(sploss.item())
            sslloss_value_reduce = misc.all_reduce_mean(sslloss.item())
            imgloss_value_reduce = misc.all_reduce_mean(imgloss.item())
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_sploss', sploss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_sslloss', sslloss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_imgloss', imgloss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        torch.cuda.empty_cache()
    #lr scheduler
    if args.lr_scheduler=='cosine':
        lr_scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def valid_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, device: torch.device, 
                    epoch: int,
                    args=None):

    model.train=False
    keys = ['noise_loss', 'loss', 'noise_psnr', 'psnr', 'noise_ssim', 'ssim', 'noise_nmse', 'nmse']
    valid_stats = {k:0 for k in keys}
    vnum = len(data_loader)

    save_folder = os.path.join(args.output_dir, 'valid_epoch{:02d}'.format(epoch))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full = data['full'].to(device, non_blocking=True)

            """ if args.downsample < 2:
                samples = samples*(mask.to(samples.device))
                ssl_masks = smasks.to(samples.device) """

            pred = model(down=samples, ssl_masks=ssl_masks, full=full, mask_ratio=0)
            
            if args.domain=='kspace':
                samples, pred, full = rifft2(samples[0,:,:,:], pred[0,:,:,:], full[0,:,:,:], permute=True) 
            elif args.domain=='img':
                samples = samples[0,:,:,:]
                pred = pred[0,:,:,:]
                full = full[0,:,:,:]
            
            #normalization [0-1]
            max = torch.max(samples)
            min = torch.min(samples)
            samples = torch.clamp((samples-min)/(max-min), min=0, max=1)
            pred = torch.clamp((pred-min)/(max-min), min=0, max=1)
            full = torch.clamp((full-min)/(max-min), min=0, max=1)
            
            #calculate psnr, ssim
            stat = calc_metrics(samples.unsqueeze(0), pred.unsqueeze(0), full.unsqueeze(0))
            for k,v in stat.items():
                valid_stats[k]+=v/vnum
            
            #image save
            if epoch//40>0 and epoch%40==0:
                if i%100==0:
                    imageio.imwrite(os.path.join(save_folder, 'ep{:02d}_concat_{:03d}.tif'.format(epoch, int(i/100))), torch.cat([samples, pred, full], dim=-1).squeeze().cpu().numpy())

    print('Validation Epoch: {} {}'.format(epoch, ', '.join(['{}: {:.3f}'.format(k,v.item()) for k,v in valid_stats.items()])))
    return valid_stats