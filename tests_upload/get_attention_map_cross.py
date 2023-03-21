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

import matplotlib.pyplot as plt

import argparse
import os
import models

from data.ixidata import IXIDataset

def forward_wrapper(attn_obj):
    def my_forward(x, complement):
        
        # x [B, HW, C]
        B_x, N_x, C_x = x.shape
        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, HW, C//num_heads]
        q = attn_obj.to_q(x).reshape(B_x, N_x, attn_obj.num_heads, C_x//attn_obj.num_heads).permute(0, 2, 1, 3)
        kv = attn_obj.to_kv(complement).reshape(B_c, N_c, 2, attn_obj.num_heads, C_c//attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)

        x = x + x_copy

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
                                    'mae_hivit_small', 'mae_hivit_base', 'hivit_small', 'hivit_base', 'himae_base', 'himae_small',
                                    'mae_alt_small', 'vit_alt_small', 'mae_cross_small', 'vit_cross_small'],
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
    parser.add_argument('--output_dir', default='./tests/attention_cross')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint. ex)1023_mae/checkpoint-best.pth')
    parser.add_argument('--save_num', default=100, type=int, help='0 is saving all images, otherwise saving only that number of images')
    parser.add_argument('--patch_direction', type=str, nargs='+', default='ro', choices=['ro', 'pe', '2d'], help='1D patch direction: readout or phase-encoding')
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
        dataset, batch_size=1, num_workers=10, pin_memory=True, drop_last=True, shuffle=True
    )
    in_chans = 2 if args.domain=='kspace' else 1
    model = models.__dict__[args.model](ssl=args.ssl, patch_size=args.patch_size, in_chans=in_chans, domain=args.domain, patch_direction=args.patch_direction)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print('Start test.. best epoch: {}'.format(checkpoint['epoch']))

    model.train=False
    model.blocks.layers[0][0].fn.attn.forward = forward_wrapper(model.blocks.layers[0][0].fn.attn)
    model.blocks.layers[0][1].fn.attn.forward = forward_wrapper(model.blocks.layers[0][1].fn.attn)
    model.blocks.layers[1][0].fn.attn.forward = forward_wrapper(model.blocks.layers[1][0].fn.attn)
    model.blocks.layers[1][1].fn.attn.forward = forward_wrapper(model.blocks.layers[1][1].fn.attn)
    model.decoder_blocks.layers[0][0].fn.attn.forward = forward_wrapper(model.decoder_blocks.layers[0][0].fn.attn)
    model.decoder_blocks.layers[0][1].fn.attn.forward = forward_wrapper(model.decoder_blocks.layers[0][1].fn.attn)
    model.decoder_blocks.layers[1][0].fn.attn.forward = forward_wrapper(model.decoder_blocks.layers[1][0].fn.attn)
    model.decoder_blocks.layers[1][1].fn.attn.forward = forward_wrapper(model.decoder_blocks.layers[1][1].fn.attn)

    with torch.no_grad():
        for i,data in enumerate(data_loader):
            samples = data['down'].to(device, non_blocking=True)
            ssl_masks = data['mask'].to(device, non_blocking=True) # 0 is keep, 1 is remove
            full_samples = data['full'].to(device, non_blocking=True)

            y = model(samples, ssl_masks, full_samples)
            attn_pe = [] #8x(1,16,257,257)
            attn_2d = []
            attn_pe.append(model.blocks.layers[0][0].fn.attn.attn_map.detach().mean(dim=1))
            attn_2d.append(model.blocks.layers[0][1].fn.attn.attn_map.detach().mean(dim=1))
            attn_pe.append(model.blocks.layers[1][0].fn.attn.attn_map.detach().mean(dim=1))
            attn_2d.append(model.blocks.layers[1][1].fn.attn.attn_map.detach().mean(dim=1))
            attn_pe.append(model.decoder_blocks.layers[0][0].fn.attn.attn_map.detach().mean(dim=1))
            attn_2d.append(model.decoder_blocks.layers[0][1].fn.attn.attn_map.detach().mean(dim=1))
            attn_pe.append(model.decoder_blocks.layers[1][0].fn.attn.attn_map.detach().mean(dim=1))
            attn_2d.append(model.decoder_blocks.layers[1][1].fn.attn.attn_map.detach().mean(dim=1))

            attn_pe_map = torch.cat(attn_pe, dim=0) #4 257 514
            attn_2d_map = torch.cat(attn_2d, dim=0) #4 257 514
            # print(attn_map.shape) #8 16 257 257
            # average the attention weights across all heads
            # attn_map = torch.mean(attn_map, dim=1) #8,1,257,257
            print(attn_pe_map.shape) #4 257 514

            #add identity matrix, account for residual connection
            res_map = torch.cat([torch.eye(attn_pe_map.size(1)), torch.zeros(attn_pe_map.size(1), attn_pe_map.size(1))], dim=-1).to(args.device)
            attn_pe_map = res_map+attn_pe_map
            attn_pe_map = attn_pe_map/attn_pe_map.sum(dim=-1).unsqueeze(-1)

            attn_2d_map = res_map+attn_2d_map
            attn_2d_map = attn_2d_map/attn_2d_map.sum(dim=-1).unsqueeze(-1)

            #pe attention
            attn_pe_map1 = attn_pe_map[:,1:,1:257] #4 256 256
            attn_pe_map2 = attn_pe_map[:,1:,258:] #4 256 256
            print(attn_pe_map1.shape) 

            patch_attn_pe1 = attn_pe_map1.reshape(4,-1,1,1,256).repeat(1,1,1,16,1) #4 256 1 16 256
            patch_attn_pe2 = attn_pe_map2.reshape(4,-1,1,16,16).repeat(1,1,1,1,1) #4 256 1 16 16
            print(patch_attn_pe1.shape, patch_attn_pe2.shape)

            for j in range(4):
                block_dir1 = os.path.join(args.output_dir, 'pe2pe_block_{:d}'.format(j+1))
                block_dir2 = os.path.join(args.output_dir, 'pe22d_block_{:d}'.format(j+1))
                if not os.path.exists(block_dir1): 
                    os.mkdir(block_dir1)
                if not os.path.exists(block_dir2): 
                    os.mkdir(block_dir2)
                for i in range(256):
                    img_pe = patch_attn_pe1[j,i,:,:,:] #1 16 256
                    img_2d = patch_attn_pe2[j,i,:,:,:] #1 16 16
                    h_pe=0
                    w_pe=i
                    h_2d=i//16
                    w_2d=i%16
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(torch.clip(img_pe[:,:,:], min=0, max=0.1), normalize=True, scale_each=True), 
                        os.path.join(block_dir1, 'pe2pe_block{}_{}x{}.png'.format(j,h_pe,w_pe)))
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(torch.clip(img_2d[:,:,:], min=0, max=0.1), normalize=True, scale_each=True), 
                        os.path.join(block_dir2, 'pe22d_block{}_{}x{}.png'.format(j,h_2d,w_2d)))

            #2d attention
            attn_2d_map1 = attn_2d_map[:,1:,1:257] #4 256 256
            attn_2d_map2 = attn_2d_map[:,1:,258:] #4 256 256
            print(attn_2d_map1.shape) 

            patch_attn_2d1 = attn_2d_map1.reshape(4,-1,1,16,16).repeat(1,1,1,1,1) #4 256 1 16 16
            patch_attn_2d2 = attn_2d_map2.reshape(4,-1,1,1,256).repeat(1,1,1,16,1) #4 256 1 16 256
            print(patch_attn_2d1.shape, patch_attn_2d2.shape)

            for j in range(4):
                block_dir1 = os.path.join(args.output_dir, '2d22d_block_{:d}'.format(j+1))
                block_dir2 = os.path.join(args.output_dir, '2d2pe_block_{:d}'.format(j+1))
                if not os.path.exists(block_dir1): 
                    os.mkdir(block_dir1)
                if not os.path.exists(block_dir2): 
                    os.mkdir(block_dir2)
                for i in range(256):
                    img_2d = patch_attn_2d1[j,i,:,:,:] #1 16 256
                    img_pe = patch_attn_2d2[j,i,:,:,:] #1 16 16
                    h_pe=0
                    w_pe=i
                    h_2d=i//16
                    w_2d=i%16
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(torch.clip(img_2d[:,:,:], min=0, max=0.1), normalize=True, scale_each=True), 
                        os.path.join(block_dir1, '2d22d_block{}_{}x{}.png'.format(j,h_2d,w_2d)))
                    torchvision.utils.save_image(
                        torchvision.utils.make_grid(torch.clip(img_pe[:,:,:], min=0, max=0.1), normalize=True, scale_each=True), 
                        os.path.join(block_dir2, '2d2pe_block{}_{}x{}.png'.format(j,h_pe,w_pe)))

            exit()

