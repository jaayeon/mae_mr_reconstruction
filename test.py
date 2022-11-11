from email.mime import image
from random import sample
import torch
import os
import argparse
import numpy as np
import imageio
from pathlib import Path
import json

from data.ixidata import IXIDataset
from util.mri_tools import rifft2
from util.metric import calc_metrics

from torch.utils.tensorboard import SummaryWriter

import models_mae_1d
import models_mae
# import util.misc as misc

def get_args_parser():
    parser = argparse.ArgumentParser('MAE test', add_help=False)
    # Model parameters
    # Model parameters
    parser.add_argument('--model', default='mae2d_small', type=str, 
                        choices=['mae2d_large', 'mae2d_base', 'mae2d_small', 'mae1d_large', 'mae1d_base', 'mae1d_small',
                                    'vit2d_large', 'vit2d_base', 'vit2d_small', 'vit1d_large', 'vit1d_base', 'vit1d_small'],
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=256, type=int, #default 224
                        help='images input size')
    parser.add_argument('--ssl', action='store_true',
                        help='make two different augmentation for each data, and calculate self supervised loss')

    # Data Preprocessing
    parser.add_argument('--down', default='uniform', choices=['uniform', 'random'], 
                        help='method of constructing undersampled data')
    parser.add_argument('--downsample', type=int, default=4, help='downsampling factor of original data')
    parser.add_argument('--low_freq_ratio', type=float, default=0.7, help='ratio of low frequency lines in undersampled data')

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
    parser.add_argument('--save_num', default=1000, type=int, help='0 is saving all images, otherwise saving only that number of images')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')

    return parser


def main(args):
    # misc.init_distributed_mode(args)

    args.resume = os.path.join(args.output_dir, args.resume)
    args.output_dir = '/'.join(args.resume.split('/')[:-1])
    base = os.path.basename(args.output_dir)

    print('output dir: {}'.format(args.output_dir))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dataset = IXIDataset(args, mode='test')
    # global_rank = misc.get_rank()

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=10, pin_memory=True, drop_last=False
    )

    if '1d' not in args.model:
        model = models_mae.__dict__[args.model](ssl=args.ssl)
    else:
        model = models_mae_1d.__dict__[args.model](ssl=args.ssl)

    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print('Start test.. best epoch: {}'.format(checkpoint['epoch']))

    model.train=False
    keys = ['noise_loss', 'loss', 'noise_psnr', 'psnr', 'psnr_dc', 'noise_ssim', 'ssim', 'ssim_dc', 'noise_nmse', 'nmse']
    test_stats = {k:0 for k in keys}
    tnum = len(data_loader) if len(data_loader)<args.save_num else args.save_num

    save_folder = os.path.join(args.output_dir, 'test')
    save_concat = os.path.join(save_folder, 'test_concat')
    save_pred = os.path.join(save_folder, 'test_pred')
    save_pred_dc = os.path.join(save_folder, 'test_pred_dc')
    save_concat_kspace = os.path.join(save_folder, 'test_concat_kspace')


    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_concat):
        os.mkdir(save_concat)
    if not os.path.exists(save_pred_dc):
        os.mkdir(save_pred_dc)
    if not os.path.exists(save_pred):
        os.mkdir(save_pred)
    if not os.path.exists(save_concat_kspace):
        os.mkdir(save_concat_kspace)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full_samples = data['full'].to(device, non_blocking=True)
            _, _, pred, _  = model(samples, ssl_masks, full_samples)


            # Data consistency
            pred = torch.clamp(pred, min=-1, max=1)
            pred_dc = samples + pred*ssl_masks

            isamples, ipred, ipred_dc, ifull = rifft2(samples[0,:,:,:], pred[0,:,:,:], pred_dc[0,:,:,:], full_samples[0,:,:,:], permute=True)

            concat_kspace = torch.cat([samples[:,0,:,:], pred[:,0,:,:], pred_dc[:,0,:,:], full_samples[:,0,:,:]], dim=-1).squeeze(0)
            #normalization [0-1]
            max = torch.max(isamples)
            min = torch.min(isamples)
            isamples = torch.clamp((isamples-min)/(max-min), min=0, max=1)
            ipred = torch.clamp((ipred-min)/(max-min), min=0, max=1)
            ipred_dc = torch.clamp((ipred_dc-min)/(max-min), min=0, max=1)
            ifull = torch.clamp((ifull-min)/(max-min), min=0, max=1)

            #calculate psnr, ssim
            stat = calc_metrics(isamples.unsqueeze(0), ipred.unsqueeze(0), ifull.unsqueeze(0))
            stat_dc = calc_metrics(isamples.unsqueeze(0), ipred_dc.unsqueeze(0), ifull.unsqueeze(0))
            for k,v in stat.items():
                test_stats[k]+=v/tnum
            test_stats['psnr_dc'] += stat_dc['psnr']/tnum
            test_stats['ssim_dc'] += stat_dc['ssim']/tnum

            imageio.imwrite(os.path.join(save_concat_kspace, 'concat_kspace_{:03d}.tif'.format(int(i))), concat_kspace.cpu().numpy())
            imageio.imwrite(os.path.join(save_pred_dc, 'preddc_{:03d}.tif'.format(int(i))), ipred_dc.squeeze().cpu().numpy())
            imageio.imwrite(os.path.join(save_pred, 'pred_{:03d}.tif'.format(int(i))), ipred.squeeze().cpu().numpy())
            imageio.imwrite(os.path.join(save_concat, 'concat_{:03d}.tif'.format(int(i))), torch.cat([isamples, ipred, ipred_dc, ifull], dim=-1).squeeze().cpu().numpy())

            if args.save_num==i-1:
                break

    with open(os.path.join(save_folder, 'test_log.txt'), mode='a', encoding="utf-8") as f:
        f.write(', '.join(['{}: {:.3f}'.format(k,v.item()) for k,v in test_stats.items()])+'\n')
    print('Test: {}'.format(', '.join(['{}: {:.3f}'.format(k,v.item()) for k,v in test_stats.items()])))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)