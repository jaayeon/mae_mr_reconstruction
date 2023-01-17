'''
https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
https://github.com/rwightman/pytorch-image-models/discussions/1232
https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map/notebook

https://epfml.github.io/attention-cnn/
[kspace transformer for undersampled MRI Reconstruction]; attention visualization included.
'''

import torch
import torch.nn as nn
import torchvision

import argparse
import os
import models

from data.ixidata import IXIDataset

def forward_wrapper(attn_obj):
    def my_forward(x):
        B,N,C = x.shape
        qkv = attn_obj.qkv(x).reshape(B,N,3,attn_obj.num_heads, C//attn_obj.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)

        attn = (q @ k.transpose(-2,-1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:,:,0,2:]

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Get attention map', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='mae2d_small', type=str, 
                        choices=['mae2d_optim', 'mae2d_large', 'mae2d_base', 'mae2d_small', 'mae1d_large', 'mae1d_base', 'mae1d_small',
                                    'vit2d_large', 'vit2d_base', 'vit2d_small', 'vit1d_large', 'vit1d_base', 'vit1d_small',
                                    'mae_hivit_small', 'mae_hivit_base', 'hivit_small', 'hivit_base', 'himae_base', 'himae_small'],
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int)
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
    parser.add_argument('--domain', default='kspace', choices=['kspace', 'img'])

    # Attention
    parser.add_argument('--output_dir', default='./tests/attention')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint. ex)1023_mae/checkpoint-best.pth')
    parser.add_argument('--save_num', default=100, type=int, help='0 is saving all images, otherwise saving only that number of images')
    args = parser.parse_args()


    if not os.path.exists('./tests'):
        os.mkdir('./tests')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.resume = os.path.join('../../data/ixi/checkpoints', args.resume)

    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    dataset = IXIDataset(args, mode='test')

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=10, pin_memory=True, drop_last=True
    )
    in_chans = 2 if args.domain=='kspace' else 1
    model = models.__dict__[args.model](ssl=args.ssl, patch_size=args.patch_size, in_chans=in_chans, domain=args.domain)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print('Start test.. best epoch: {}'.format(checkpoint['epoch']))

    model.train=False
    model.decoder_blocks[-1].attn.forward = forward_wrapper(model.decoder_blocks[-1].attn)

    with torch.no_grad():
        for i,data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full_samples = data['full'].to(device, non_blocking=True)

            y = model(samples, ssl_masks, full_samples)
            attn_map = model.decoder_blocks[-1].attn.attn_map.detach()
            print(attn_map.shape) #1 16 257 257
            # average the attention weights across all heads
            attn_map = torch.mean(attn_map, dim=1)
            print(attn_map.shape) #1 257 257

            #add identity matrix, account for residual connection
            res_map = torch.eye(attn_map.size(1)).to(args.device)
            attn_map = res_map+attn_map
            attn_map = attn_map/attn_map.sum(dim=-1).unsqueeze(-1)

            attn_map = attn_map[0,:-1,:-1]
            patch_attn = attn_map.reshape(-1,1,16,16)
            torchvision.utils.save_image(
                    torchvision.utils.make_grid(patch_attn, normalize=True, scale_each=True, nrow=16, pad_value=0.5, padding=1), 
                    os.path.join(args.output_dir, 'attn.png'))
            for i in range(256):
                img = patch_attn[i,:,:,:]
                h=i//16
                w=i%16
                torchvision.utils.save_image(
                    torchvision.utils.make_grid(img, normalize=True, scale_each=True), 
                    os.path.join(args.output_dir, 'attn_patch_{}x{}.png'.format(h,w)))
                # torchvision.utils.save_image(img, os.path.join(args.output_dir, 'attn_patch_{}x{}.png'.format(h,w)))
            exit()