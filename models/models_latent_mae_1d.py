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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from timm.models.vision_transformer import Block

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


class PatchEmbed(nn.Module):
    def __init__(self, patch_direction='ro', img_size=256, in_chans=2, embed_dim=768):
        super().__init__()
        """
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        self.pd = patch_direction
        self.img_size = img_size
        self.in_chans = in_chans
        self.proj = nn.Linear(in_chans*img_size, embed_dim)

        self.num_patches = img_size

    def forward(self, imgs):
        if self.pd=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
        elif self.pd=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
        x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
        x = self.proj(x)
        return x



class LatentMaskedAutoencoderViT1d(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vit, patch_direction='ro', domain='kspace', img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mae=True, ssl=False,
                 num_low_freqs=None):
        super().__init__()

        self.in_chans = in_chans
        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if type(patch_direction)==list and len(patch_direction)==1:
            patch_direction=patch_direction[0]

        # self.head = Head(in_chans, 32)
        self.patch_embed = PatchEmbed(patch_direction, img_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans*img_size, bias=True) # decoder to patch
        # self.decoder_pred = nn.Conv2d(decoder_embed_dim//img_size, in_chans, 1)
        # self.predictor = nn.Sequential(nn.Linear(in_chans*img_size, 128, bias=True),
        #                                 nn.GELU(),
        #                                 nn.Linear(128, in_chans*img_size))
        # --------------------------------------------------------------------------

        self.discriminator = Discriminator(1)

        self.wrap_blocks()

        self.norm_pix_loss = norm_pix_loss
        self.ssl = ssl
        self.mae = mae
        self.mask_center = mask_center
        self.train = True
        self.img_size = img_size
        self.num_low_freqs = num_low_freqs
        self.divide_loss = focal_gaussian() if divide_loss else None
        self.domain = domain
        self.in_chans = in_chans
        self.pd = patch_direction
        assert self.pd=='ro' or self.pd=='pe'

        self.depth = depth
        self.decoder_depth = decoder_depth
        self.guided_attention = guided_attention
        self.regularize_attnmap = True if regularize_attnmap else False

        # self.ploss = perceptualloss()

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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # decoder_pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
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
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        if self.pd=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
        elif self.pd=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
        x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, H, cxW)
        imgs: (N, c, H, W)
        """      
        x = x.reshape(shape=(x.shape[0], self.img_size, self.img_size, -1))
        if self.pd=='ro':
            imgs = torch.einsum('nhwc->nchw', x)
        elif self.pd=='pe':
            imgs = torch.einsum('nwhc->nchw', x)
        return imgs

    def attmap_regularization(self, reguralization='l2'):
        attn = []
        for i in range(self.depth):
            attn.append(self.blocks[i].attn.attn_map.detach().mean(dim=1))
        N,L,_ = attn[-1].shape
        attn = torch.cat(attn, dim=0)

        attn_decoder = []
        for i in range(self.decoder_depth):
            attn_decoder.append(self.decoder_blocks[i].attn.attn_map.detach().mean(dim=1))
        Nd,Ld,_ = attn_decoder[-1].shape
        attn_d = torch.cat(attn_decoder, dim=0)

        # get regularization value
        if reguralization == 'l2':
            reg = (torch.sum(attn_d**2)+torch.sum(attn**2))/(self.depth + self.decoder_depth)/N/L
        else:
            reg = 0.0
        return reg


    def random_masking(self, x, mask_ratio, ssl_masks, given_ids_shuffle=None):
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
        
        removed_index=None

        if given_ids_shuffle is not None:
            ids_shuffle = given_ids_shuffle
            flip_ids_shuffle = None
        else:
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

            if not self.mask_center:
                # get center mask start index & ending index
                start = int((L-self.num_low_freqs)/2)
                end = int((L+self.num_low_freqs)/2)

                # get overall center mask index to preserve
                ids_low_freq = torch.tensor([i for i in range(start, end)], device=x.device) #[l]
                # order randomization
                low_noise = torch.rand(len(ids_low_freq), device=x.device)
                low_ids_shuffle = torch.argsort(low_noise)
                ids_low_freq = torch.gather(ids_low_freq, dim=0, index=low_ids_shuffle)
                # only preserve 80% of center in default
                len_low_keep = int(len(ids_low_freq)*0.8) #0.8
                ids_low_keep = ids_low_freq[:len_low_keep]

                # concat center index to the front; must included & delete duplicated index
                _ids_shuffle = torch.ones(ids_shuffle.shape, device=x.device)
                for i in range(len(ids_low_keep)):
                    _ids_shuffle = _ids_shuffle==(ids_shuffle!=ids_low_keep[i]) # True : not included in ids_low_keep, accumulate the False 
                _ids_shuffle=_ids_shuffle.nonzero(as_tuple=True)[1].view(N, -1) # get True index (N,L-l)
                _ids_shuffle = torch.gather(ids_shuffle, dim=1, index=_ids_shuffle) # get elements not included in ids_low_keep
                ids_shuffle = torch.cat([ids_low_keep.repeat(N,1), _ids_shuffle], dim=1).type(torch.int64)

            if removed_index is not None:
                #concat removed index to the back & delete duplicated index
                _ids_shuffle = torch.ones(ids_shuffle.shape, device=x.device)
                for i in range(len(removed_index)):
                    _ids_shuffle = _ids_shuffle==(ids_shuffle!=removed_index[i])
                _ids_shuffle = _ids_shuffle.nonzero(as_tuple=True)[1].view(N,-1)
                _ids_shuffle = torch.gather(ids_shuffle, dim=1, index=_ids_shuffle)
                ids_shuffle = torch.cat([_ids_shuffle, removed_index.repeat(N,1)], dim=1).type(torch.int64)

            flip_ids_shuffle=None


        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(N,L*0.75's index,D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #(N,L)

        # return x_masked, mask, ids_restore, flip_ids_shuffle
        return x_masked, mask, ids_restore, ids_shuffle

    def forward_encoder(self, x, mask_ratio, ssl_masks, given_ids_shuffle=None, ema=False):
        #head 
        # x = self.head(x, ssl_masks)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.train and self.mae:
            x, mask, ids_restore, pair_ids = self.random_masking(x, mask_ratio, ssl_masks, given_ids_shuffle=given_ids_shuffle)
        else:
            mask = None
            ids_restore = None
            pair_ids = None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, pair_ids



    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if self.train and self.mae:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #(1,1,D)->(N,L*0.75,D)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> to add positional embed
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        if not torch.isfinite(x).all():
            print('anomaly detected d1')

        # apply Transformer blocks
        '''
        for i, blk in enumerate(self.decoder_blocks):
            if i > 5:
                with torch.cuda.amp.autocast(enabled=False):
                    x=blk(x.float())
            else:
                x = blk(x)
            if not torch.isfinite(x).all():
                print('anomaly detected in after {}th block'.format(i))
        '''
        for blk in self.decoder_blocks:
            x = blk(x)
        

        if not torch.isfinite(x).all():
            print('anomaly detected d2')
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        # x = self.decoder_pred(x)
        # x = self.unpatchify(x)
        x = self.decoder_pred(x)
        x = self.unpatchify(x)

        return x

    def forward_loss(self, imgs, pred, mask, ssl_masks, full=None):
        """
        imgs: [N, 2, H, W]
        pred: [N, L, p*p*2]
        mask: [N, L], 0 is keep, 1 is remove, 
        ssl_masks: [N, 1, H, W] 0 is keep, 1 is remove
        sp_masks: [N, 1, H, W] 0 is remove, 1 is keep
        """
        N=pred.size(0)

        sp_masks = 1-ssl_masks
        # sp_masks = self.patchify(sp_masks)
        if full is not None:
            # target = self.patchify(full)
            target = full
        else:
            # target = self.patchify(imgs)
            target = imgs
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # loss = (pred - target) ** 2
        loss = torch.abs(pred - target)
        if self.ssl: # calculate loss in only acquired data 
            loss = loss*sp_masks
        '''
        if self.divide_loss is not None:
            # divide_loss = self.patchify(self.divide_loss)
            loss = loss*divide_loss.to(loss.device)
        '''
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if self.ssl:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = loss.sum() / N  # mean loss on every patches
        
        return loss

    def bce_discr_loss(self, fake, real):
        return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()
    
    def hinge_discr_loss(self, fake, real):
        return (F.relu(1 + fake) + F.relu(1 - real)).mean()
    
    def hinge_gen_loss(self, fake):
        return -fake.mean()


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
        latent, mask, ids_restore, ids_shuffle = self.forward_encoder(down, mask_ratio, ssl_masks)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        if self.train and self.regularize_attnmap:
            reg = self.attmap_regularization()
        else:
            reg = 0.0

        #dc layer
        pred = down + pred*ssl_masks

        #ifft
        predimg, fullimg = rifft2(pred, full, permute=True)
        maxnum = torch.max(fullimg)
        minnum = torch.min(fullimg)
        predimg = (predimg-minnum)/(maxnum-minnum+1e-08)
        fullimg = (fullimg-minnum)/(maxnum-minnum+1e-08)
        #predimg = normalize(predimg)
        #fullimg = normalize(fullimg)


        if self.train:
            #reconstruction loss
            loss = self.forward_kspace_loss(down, pred, full=full) #mask: 0 is keep, 1 is remove
            imgloss = self.forward_img_loss(predimg, fullimg)

            #generator loss
            gen_loss = self.hinge_gen_loss(self.discriminator(predimg))

            #discriminator loss
            pred_discr, full_discr = map(self.discriminator, (predimg.detach(), fullimg.detach()))
            discr_loss = self.hinge_discr_loss(pred_discr, full_discr)
            gp = gradient_penalty(full_discr, full_discr)
            discr_loss += gp


            return loss, imgloss, torch.tensor([0], device=loss.device), gen_loss+discr_loss
        else: #test
            return pred

class AdversarialViT1d(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_direction='ro', domain='kspace', img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 num_low_freqs=None, use_ema=True, **args):
        super().__init__()

        self.in_chans = in_chans
        self.img_size = img_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if type(patch_direction)==list and len(patch_direction)==1:
            patch_direction=patch_direction[0]

        # self.head = Head(in_chans, 32)
        self.patch_embed = PatchEmbed(patch_direction, img_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None -> LayerScale=None..?
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans*img_size, bias=True) # decoder to patch

        # --------------------------------------------------------------------------

        # self.discriminator = Discriminator(1)
        self.use_ema = use_ema
        if use_ema:
            self.ema_blocks = EMA(self.blocks, update_after_step=0, update_every=1)
            self.ema_blocks.eval() 

        self.wrap_blocks()

        self.train = True
        self.img_size = img_size
        self.num_low_freqs = num_low_freqs

        self.domain = domain
        self.in_chans = in_chans
        self.pd = patch_direction
        assert self.pd=='ro' or self.pd=='pe'

        self.depth = depth
        self.decoder_depth = decoder_depth

        self.initialize_weights()


    def wrap_blocks(self):
        self.blocks[0].attn.forward = forward_wrapper(self.blocks[0].attn)
        self.blocks[1].attn.forward = forward_wrapper(self.blocks[1].attn)
        self.blocks[2].attn.forward = forward_wrapper(self.blocks[2].attn)
        # self.blocks[3].attn.forward = forward_wrapper(self.blocks[3].attn)
        self.decoder_blocks[0].attn.forward = forward_wrapper(self.decoder_blocks[0].attn)
        self.decoder_blocks[1].attn.forward = forward_wrapper(self.decoder_blocks[1].attn)
        self.decoder_blocks[2].attn.forward = forward_wrapper(self.decoder_blocks[2].attn)
        # self.decoder_blocks[3].attn.forward = forward_wrapper(self.decoder_blocks[3].attn)


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # decoder_pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        if self.pd=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
        elif self.pd=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
        x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, H, cxW)
        imgs: (N, c, H, W)
        """      
        x = x.reshape(shape=(x.shape[0], self.img_size, self.img_size, -1))
        if self.pd=='ro':
            imgs = torch.einsum('nhwc->nchw', x)
        elif self.pd=='pe':
            imgs = torch.einsum('nwhc->nchw', x)
        return imgs

    def random_masking(self, x, mask_ratio, ssl_masks):
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

        if ssl_masks is not None:
            assert torch.sum(ssl_masks)!=0
            len_keep = int(L-torch.sum(ssl_masks[0,0,:,0]))
            removed_index = ssl_masks[0,0,:,0].nonzero(as_tuple=True)[0] #1: unscanned, 0: scanned
            ids_shuffle = torch.arange(L).repeat(N,1)
            for i in range(len(removed_index)):
                ids_shuffle = ids_shuffle[:,ids_shuffle[0]!=removed_index[i]]
            ids_shuffle = torch.cat([ids_shuffle, removed_index.repeat(N,1)], 1) #N,L


        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(N,L*0.75's index,D)

        # generate the binary mask: 0 is keep, 1 is remove
        downmask = torch.zeros([N, L], device=x.device)
        downmask[:, :len_keep] = 1
        # unshuffle to get the binary downmask
        downmask = torch.gather(downmask, dim=1, index=ids_restore) #(N,L) unmasked region: 1, masked region: 0

        # return x_masked, downmask, ids_restore, flip_ids_shuffle
        return x_masked, downmask, ids_restore, ids_shuffle

    def forward_encoder(self, x, mask_ratio=0, ssl_masks=None, ema=False):
        #head 
        # x = self.head(x, ssl_masks)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # apply Transformer blocks
        if ema:
            x, downmask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio, ssl_masks=ssl_masks)
            for blk in self.ema_blocks:
                x = blk(x)
        else:
            downmask, ids_restore, ids_shuffle = None, None, None
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)

        return x, downmask, ids_restore, ids_shuffle


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

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        # x = self.decoder_pred(x)
        # x = self.unpatchify(x)
        x = self.decoder_pred(x)
        x = self.unpatchify(x)

        return x

    def bce_discr_loss(self, fake, real):
        return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()
    
    def hinge_discr_loss(self, fake, real):
        return (F.relu(1 + fake) + F.relu(1 - real)).mean()
    
    def hinge_gen_loss(self, fake):
        return -fake.mean()

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
    
    def forward_latent_loss(self, latent1, latent2, masks):



    def forward(self, down, ssl_masks, full, input_down=False, mask_ratio=0.25, **args):
        if not input_down: #full to full
            down = full
        latent= self.forward_encoder(full)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]

        if self.ema:
            with torch.no_grad():
                latent_ema = self.forward_encoder(full, mask_ratio=mask_ratio, ema=True)
            
        #dc layer
        #pred = down + pred*ssl_masks

        #ifft
        with torch.cuda.amp.autocast(enabled=False):
            predimg, fullimg = rifft2(pred.float(), full.float(), permute=True)
            maxnum = torch.max(fullimg)
            minnum = torch.min(fullimg)
            predimg = (predimg-minnum)/(maxnum-minnum+1e-08)
            fullimg = (fullimg-minnum)/(maxnum-minnum+1e-08)

        if self.train:
            #reconstruction loss
            loss = self.forward_kspace_loss(pred, full) #mask: 0 is keep, 1 is remove
            imgloss = self.forward_img_loss(predimg, fullimg)

            # #generator loss
            # gen_loss = self.hinge_gen_loss(self.discriminator(predimg))

            # #discriminator loss
            # pred_discr, full_discr = map(self.discriminator, (predimg.detach(), fullimg.detach()))
            # discr_loss = self.hinge_discr_loss(pred_discr, full_discr)
            # gp = gradient_penalty(full_discr, full_discr)
            # discr_loss += gp


            # return loss, imgloss, torch.tensor([0], device=loss.device), gen_loss+discr_loss
            return loss, imgloss, torch.tensor([0], device=loss.device), torch.tensor([0], device=loss.device)
        else: #test
            return pred


def mae_1d_large_10_1024(**kwargs):
    model = LatentMaskedAutoencoderViT1d(
        embed_dim=1024, depth=8, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_1d_base_8_768(**kwargs):
    model = LatentMaskedAutoencoderViT1d(
        embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_1d_small_6_768(**kwargs):
    model = LatentMaskedAutoencoderViT1d(
        embed_dim=768, depth=4, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_1d_large_8_1024(**kwargs):
    model = AdversarialViT1d(
        embed_dim=1024, depth=8, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)
    return model

def vit_1d_base_6_768(**kwargs):
    model = AdversarialViT1d(
        embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_1d_small_3_768(**kwargs):
    model = AdversarialViT1d(
        embed_dim=768, depth=3, num_heads=12,
        decoder_embed_dim=768, decoder_depth=3, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

lmae1d_large = mae_1d_large_10_1024
lmae1d_base = mae_1d_base_8_768
lmae1d_small =  mae_1d_small_6_768
lvit1d_large = vit_1d_large_8_1024
lvit1d_base = vit_1d_base_6_768
lvit1d_small =  vit_1d_small_3_768