# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from unittest.loader import VALID_MODULE_NAME
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import timm

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_mae_1d

from engine_pretrain import train_one_epoch, valid_one_epoch

from data.ixidata import IXIDataset
# from data.fmridata import FMRIDataset
from util.mri_tools import rifft2
from util.metric import calc_metrics


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae2d_large', type=str, 
                        choices=['mae2d_large', 'mae2d_base', 'mae2d_small', 'mae1d_large', 'mae1d_base', 'mae1d_small',
                                    'vit2d_large', 'vit2d_base', 'vit2d_small', 'vit1d_large', 'vit1d_base', 'vit1d_small'],
                        metavar='MODEL', help='Name of model to train')


    parser.add_argument('--input_size', default=256, type=int, #default 224
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--ssl', action='store_true',
                        help='make two different augmentation for each data, and calculate self supervised loss')
    parser.add_argument('--ssl_weight', type=float, default=1, help='weight of ssl loss related to sp_loss')
    parser.add_argument('--img_weight', type=float, default=0.1, help='weight of img loss in spatial domain')
    parser.add_argument('--divide_loss', action='store_true', 
                        help='to maximize the entropy, to balance the energy, divide exponential term to each pixel loss')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Data Preprocessing
    parser.add_argument('--down', default='uniform', choices=['uniform', 'random'], 
                        help='method of constructing undersampled data')
    parser.add_argument('--downsample', type=int, default=4, help='downsampling factor of original data')
    parser.add_argument('--v_downsample', type=int, default=4, help='downsampling factor of validation data')
    parser.add_argument('--low_freq_ratio', type=float, default=0.7, help='ratio of low frequency lines in undersampled data')
    parser.add_argument('--no_center_mask', action='store_true', help='preserving center in kspace from random_masking')

    # Dataset parameters
    parser.add_argument('--data_path', default='../../data/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='ixi', choices=['ixi', 'fastmri'])

    # Learning
    parser.add_argument('--output_dir', default='../../data/ixi/checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint. ex)1023_mae/checkpoint-best.pth')
    parser.add_argument('--note', default='base', type=str, help='add to checkpoint base name')
    parser.add_argument('--detect_anomaly', action='store_true', 
                        help='torch.autograd.set_detect_anomaly(true), but very slow (7~8 times)')
    parser.add_argument('--autocast', action='store_true', 
                        help='set torch.cuda.amp.autocast(): float32 -> float16. 0.6 faster, but sth cause nan value...')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # set checkpoint saving directory
    dt = datetime.datetime.now()
    if not args.resume:
        base = '{}_{}'.format(dt.strftime('%m%d'), args.note)
        args.output_dir = os.path.join(args.output_dir, base)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    else:
        args.resume = os.path.join(args.output_dir, args.resume)
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
        base = os.path.basename(args.output_dir)

    print('output dir: {}'.format(args.output_dir))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset
    dataset_train = IXIDataset(args, mode='train')
    dataset_valid = IXIDataset(args, mode='valid')

    global_rank = misc.get_rank()
    if args.distributed:
        num_tasks = misc.get_world_size()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, shuffle=False, 
        batch_size=1, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_mem, 
        drop_last=False
    )
    num_low_freqs = 44 if dataset_valid.num_low_freqs>44 else dataset_valid.num_low_freqs
    # define the model
    if '1d' not in args.model:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, ssl=args.ssl, 
                                            no_center_mask=args.no_center_mask, 
                                            num_low_freqs=num_low_freqs,
                                            divide_loss=args.divide_loss)
    else:
        model = models_mae_1d.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, ssl=args.ssl, 
                                            no_center_mask=args.no_center_mask, 
                                            num_low_freqs=num_low_freqs,
                                            divide_loss=args.divide_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, weight_decay=args.weight_decay) #add_weight_decay -> param_groups_weight_decay
    # return of param_groups_weight_decay: 
    # [{'params': no_decay, 'weight_decay': 0.},
    # {'params': decay, 'weight_decay': weight_decay}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    print(optimizer)
    loss_scaler = NativeScaler()
    best_psnr = 0.0

    #resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        train_log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "{}_log_train.txt".format(base)), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_log_stats) + "\n")
        

        # validation
        valid_stats = valid_one_epoch(
            model, data_loader_valid, 
            device, epoch, args=args
        )
        valid_log_stats = {**{f'valid_{k}': v.item() for k,v in valid_stats.items()}, 'epoch':epoch,}

        if valid_log_stats['valid_psnr_dc']>best_psnr:
            print('Save Best Checkpoint')
            best_psnr = valid_log_stats['valid_psnr_dc']
            #save
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, best=True)

        with open(os.path.join(args.output_dir, "{}_log_valid.txt".format(base)), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(valid_log_stats)+"\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
