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
    # model.train()
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
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = data['down'].to(device, non_blocking=True)
        ssl_masks = data['mask'].to(device, non_blocking=True)
        full_samples = data['full'].to(device, non_blocking=True)
        num_low_freqs = data['num_low_freqs'][0].to(device)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, ssl_masks, mask_ratio=args.mask_ratio, num_low_freqs=num_low_freqs)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # loss.backward()
        # optimizer.step()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
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
    keys = ['noise_loss', 'loss', 'noise_psnr', 'psnr', 'noise_ssim', 'ssim', 'noise_nmse', 'nmse']
    valid_stats = {k:0 for k in keys}
    vnum = len(data_loader)

    save_folder = os.path.join(args.output_dir, 'valid_epoch{:03d}'.format(epoch))
    save_concat_folder = os.path.join(save_folder, 'epoch{:03d}_concat'.format(epoch))
    save_down_folder = os.path.join(save_folder, 'epoch{:03d}_down'.format(epoch))
    save_pred_folder = os.path.join(save_folder, 'epoch{:03d}_pred'.format(epoch))
    save_full_folder = os.path.join(save_folder, 'epoch{:03d}_full'.format(epoch))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_concat_folder):
        os.mkdir(save_concat_folder)
    if not os.path.exists(save_down_folder):
        os.mkdir(save_down_folder)
    if not os.path.exists(save_pred_folder):
        os.mkdir(save_pred_folder)
    if not os.path.exists(save_full_folder):
        os.mkdir(save_full_folder)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full_samples = data['full'].to(device, non_blocking=True)
            _, pred, _  = model(samples, ssl_masks)

            # Data consistency
            pred = torch.clamp(pred, min=-1, max=1)
            pred = samples + pred*ssl_masks
            
            samples, pred, full = rifft2(samples[0,:,:,:], pred[0,:,:,:], full_samples[0,:,:,:], permute=True) 
            
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
            if i%50==0:
                imageio.imwrite(os.path.join(save_down_folder, 'ep{:03d}_down_{:03d}.tif'.format(epoch, int(i/50))), samples.squeeze().cpu().numpy())
                imageio.imwrite(os.path.join(save_pred_folder, 'ep{:03d}_pred_{:03d}.tif'.format(epoch, int(i/50))), pred.squeeze().cpu().numpy())
                imageio.imwrite(os.path.join(save_full_folder, 'ep{:03d}_full_{:03d}.tif'.format(epoch, int(i/50))), full.squeeze().cpu().numpy())
                imageio.imwrite(os.path.join(save_concat_folder, 'ep{:03d}_concat_{:03d}.tif'.format(epoch, int(i/50))), torch.cat([samples, pred, full], dim=-1).squeeze().cpu().numpy())

    print('Validation Epoch: {} {}'.format(epoch, ', '.join(['{}: {:.3f}'.format(k,v.item()) for k,v in valid_stats.items()])))
    return valid_stats


