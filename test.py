import torch
import os
import argparse
import numpy as np

from data.ixidata import IXIDataset
from util.mri_tools import rifft2
from util.metric import calc_metrics

from torch.utils.tensorboard import SummaryWriter

import models_mae
import util.misc as misc

def get_args_parse():
    parser = argparse.ArgumentParser('MAE test', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16_uniform', type=str, metavar='MODEL',
                        help='Name of model to train')
                        #mae_vit_base_patch16
                        #mae_vit_large_patch16
                        #mae_vit_huge_patch14
                        #mae_vit_base_patch16_uniform

    parser.add_argument('--input_size', default=256, type=int, #default 224
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--ssl_loss', action='store_true',
                        help='make two different augmentation for each data, and calculate self supervised loss')
    parser.add_argument('--ssl_weight', type=float, default=1, help='weight of ssl loss related to sp_loss')
    


    # Data Preprocessing
    parser.add_argument('--down', default='uniform', choices=['uniform', 'random'], 
                        help='method of constructing undersampled data')
    parser.add_argument('--downsample', type=int, default=2, help='downsampling factor of original data')
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

    return parser


def main(args):
    misc.init_distributed_mode(args)

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

    dataset = IXIDataset(args, mode='test')
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=10, pin_memory=True, drop_last=False
    )

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, ssl_loss=args.ssl_loss, 
                                            ssl_weight=args.ssl_weight, no_center_mask=args.no_center_mask)

    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print('Start test.. best epoch: {}'.format(checkpoint['epoch']))

    model.train=False
    keys = ['noise_loss', 'loss', 'noise_psnr', 'psnr', 'noise_ssim', 'ssim', 'noise_nmse', 'nmse']
    test_stats = {k:0 for k in keys}
    tnum = len(data_loader)

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
            _, pred, _  = model(samples, ssl_masks, coverall=True)


            # Data consistency
            pred = torch.clamp(pred, min=-1, max=1)
            pred = samples + pred*ssl_masks