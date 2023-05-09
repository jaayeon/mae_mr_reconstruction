# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from cgi import print_arguments
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from timm.models.vision_transformer import PatchEmbed, Block

from einops import rearrange, reduce, repeat
from ema_pytorch import EMA

from util.pos_embed import get_2d_sincos_pos_embed, focal_gaussian, get_1d_sincos_pos_embed
from util.mri_tools import rifft2, rfft2, normalize
from util.vggloss import perceptualloss

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


def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, input_size=80):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        use_bias = norm_layer!=nn.BatchNorm2d
        # use_bias=True
        kw=4 #kernel size
        padw=1 #padding
        sequence=[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult=1
        nf_mult_prev=1
        for n in range(1,n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev=nf_mult
        nf_mult=min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf*nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input, param=[0.5,1]):
        input = self.random_crop(input)

        return self.model(input)

    def random_crop(self, x):
        size = x.size()[-1] #b,c,h,w
        if size == self.input_size:
            return x
        rh = random.randint(0, size-self.input_size)
        rw = random.randint(0, size-self.input_size)
        return x[:,:,rh:rh+self.input_size, rw:rw+self.input_size]



class EMAMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_direction=None, domain='kspace', img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16, ssl=False,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mae=True, mask_center=False, 
                 num_low_freqs=None):
        super().__init__()

        self.in_chans = in_chans
        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------
        if mae:
            self.ema_blocks = nn.ModuleList([EMA(self.blocks[i], update_after_step=0, update_every=1) for i in range(depth)])
            self.ema_blocks.eval()
        
        self.wrap_blocks()

        # assert ssl==mae
        self.mae = mae
        self.mask_center = mask_center
        self.train = True
        self.img_size = img_size
        self.num_low_freqs = num_low_freqs
        self.domain = domain
        self.in_chans = in_chans

        self.depth = depth
        self.decoder_depth = decoder_depth

        self.initialize_weights()


    def wrap_blocks(self):
        self.blocks[0].attn.forward = forward_wrapper(self.blocks[0].attn)
        self.blocks[1].attn.forward = forward_wrapper(self.blocks[1].attn)
        self.blocks[2].attn.forward = forward_wrapper(self.blocks[2].attn)
        self.blocks[3].attn.forward = forward_wrapper(self.blocks[3].attn)
        self.decoder_blocks[0].attn.forward = forward_wrapper(self.decoder_blocks[0].attn)
        self.decoder_blocks[1].attn.forward = forward_wrapper(self.decoder_blocks[1].attn)
        self.decoder_blocks[2].attn.forward = forward_wrapper(self.decoder_blocks[2].attn)
        self.decoder_blocks[3].attn.forward = forward_wrapper(self.decoder_blocks[3].attn)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, c, H, W)
        x: (N, L, patch_size**2 *c)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0],self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 *self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, c, H, W)
        """

        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p,self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0],self.in_chans, h * p, h * p))
        return imgs


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        ex)
            noise = N x [0.2854, 0.8221, 0.9750, 0.4495, 0.5793, 0.3031, 0.2454, 0.9205, 0.6183, 0.9235]
            ids_shuffle = N x [6, 0, 5, 3, 4, 8, 1, 7, 9, 2]
            ids_restore = N x [1, 6, 9, 3, 4, 2, 0, 7, 5, 8]
            ids_low_freq = [3, 4, 5]
            _ids_shuffle = N x [6, 0, 8, 1, 7, 9, 2]
            ids_shuffle = N x [3, 4, 5, 6, 0, 8, 1, 7, 9, 2]
            ids_keep = N x [3, 4, 5, 6]

            ssl_masks: 0 is keep, 1 is remove (N,c,h,w)
        """        
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(N,L*0.75's index,D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #(N,L)

        # rearrange
        mask_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1) #(1,1,D)->(N,L*0.75,D)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)  # no cls token
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))  # unshuffle -> to add positional embed

        return x_masked, mask


    def forward_encoder(self, x, mask_ratio=0, ema=False):
        '''
        student: 
            x=down
            mask_ratio > 0
            ema=False
            mae=T/F both possible (T: mae, F: vit)
        teacher: 
            x=full
            mask_ratio=0
            ema=True
            mae=T
        '''
        # embed patches
        x = self.patch_embed(x)

        #for train, student, mae mode
        if self.train and mask_ratio and self.mae: 
            x, mask = self.random_masking(x, mask_ratio)
        else:
            mask = None

        # add pos embed
        x = x + self.pos_embed
        
        x_stats=[]
        # apply Transformer blocks
        if ema: #teacher/full
            for blk in self.ema_blocks:
                x = blk(x)
                x_stats.append(F.layer_norm(x, x.shape[-1:]))
        else: #student/down/mask
            for blk in self.blocks:
                x = blk(x)
                x_stats.append(F.layer_norm(x, x.shape[-1:]))
        x = self.norm(x)
        x_stats.append(x)

        return x, mask, x_stats


    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed
        if not torch.isfinite(x).all():
            print('anomaly detected d1')

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        if not torch.isfinite(x).all():
            print('anomaly detected d2')
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        x = self.unpatchify(x)

        return x


    def forward_latent_loss(self, latent1, latent2, avg=True): #0 is keep, 1 is remove
        if avg: 
            latent1 = sum(latent1)/len(latent1)
            latent2 = sum(latent2)/len(latent2)
            
        N,L1,_ = latent1.shape
        # mask = ssl_masks[:,0,:,:].squeeze() # N,c,h,w --> N,h,w
        # mask = mask[:,:,0].unsqueeze(-1)

        loss = torch.abs(latent1-latent2)
        # loss = loss*mask
        loss = loss.mean(dim=-1)
        loss = loss.sum() / N

        return loss

    def forward_kspace_loss(self, down, full):
        N,_,_,_=down.shape
        kspaceloss = torch.sum(torch.abs(down-full))/N
        return kspaceloss
        
    def forward_img_loss(self, predimg, fullimg):
        N,_,_,_=predimg.shape
        imgloss = torch.sum(torch.abs(predimg-fullimg))/N
        # ploss = self.ploss(predimg, fullimg)
        # return imgloss+ploss
        return imgloss

    def forward(self, down, ssl_masks, full, mask_ratio=0.75):
        latent, mask, latent_stats = self.forward_encoder(down, mask_ratio)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]

        if self.mae:
            with torch.no_grad():
                latent_ema, _, latent_ema_stats = self.forward_encoder(full, ema=True) #no random_masking
        #dc layer
        pred = down + pred*ssl_masks 

        #ifft
        predimg, fullimg = rifft2(pred, full, permute=True)
        maxnum = torch.max(fullimg)
        minnum = torch.min(fullimg)
        predimg = (predimg-minnum)/(maxnum-minnum+1e-08)
        fullimg = (fullimg-minnum)/(maxnum-minnum+1e-08)


        if self.train:
            loss = self.forward_kspace_loss(down, full) #mask: 0 is keep, 1 is remove
            imgloss = self.forward_img_loss(predimg, fullimg)
            zero = torch.tensor([0], device=loss.device)
            if self.mae:
                latentloss = self.forward_latent_loss(latent_stats[-3:], latent_ema_stats[-3:]) #2blocks output avg loss
                return loss, imgloss, latentloss, zero
            else:
                return loss, imgloss, zero, zero
        else: #not train, not ssl
            return pred



def ema_mae_2d_large_8_1024(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=1024, depth=8, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16, 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ema_mae_2d_base_6_768(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=16, 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ema_mae_2d_small_4_768(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=768, depth=4, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def ema_vit_2d_large_8_1024(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=1024, depth=8, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs)
    return model

def ema_vit_2d_base_6_768(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs)
    return model

def ema_vit_2d_small_4_768(**kwargs):
    model = EMAMaskedAutoencoderViT(
        embed_dim=768, depth=4, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs)
    return model

ema_mae2d_large = ema_mae_2d_large_8_1024
ema_mae2d_base = ema_mae_2d_base_6_768
ema_mae2d_small =  ema_mae_2d_small_4_768
ema_vit2d_large = ema_vit_2d_large_8_1024
ema_vit2d_base = ema_vit_2d_base_6_768
ema_vit2d_small =  ema_vit_2d_small_4_768