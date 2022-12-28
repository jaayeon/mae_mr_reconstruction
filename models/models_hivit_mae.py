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

from timm.models.vision_transformer import Block #, PatchEmbed
from timm.models.vision_transformer import trunc_normal_

from .models_hivit import HiViT, PatchEmbed, PatchMerge, BlockWithRPE

from util.pos_embed import get_2d_sincos_pos_embed, focal_gaussian
from util.mri_tools import rifft2, normalize


class HiViTMaskedAutoencoder(HiViT, nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=512, depths=[2,2,10], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, hifeat=False,
                 mae=True, ssl=False, mask_center=False, num_low_freqs=None, divide_loss=False, norm_pix_loss=False):
        nn.Module.__init__(self)
        
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]
        self.hifeat = hifeat
        self.norm_pix_loss = norm_pix_loss

        embed_dim = embed_dim // 2 ** (self.num_layers-1)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features, requires_grad=False)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] += Hp - 1 
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding


        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, 
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), 
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.num_features = 7 * embed_dim if self.hifeat else embed_dim
        self.norm = norm_layer(self.num_features)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_size = patch_size
        self.decoder_embed = nn.Linear(self.num_features, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale, 
                rpe=False, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.decoder_patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.ssl = ssl
        self.mae = mae
        self.mask_center = mask_center
        self.train = True
        self.img_size = img_size
        self.num_low_freqs = num_low_freqs
        self.divide_loss = focal_gaussian() if divide_loss else None

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        Hp, Wp = self.patch_embed.patches_resolution
        pos_embed = get_2d_sincos_pos_embed(self.absolute_pos_embed.shape[-1], Hp, cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], Hp, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
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
        c=2
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, c, H, W)
        """
        c=2
        p = self.decoder_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def masking_id(self, batch_size, mask_ratio):
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
        """
        N, L = batch_size, self.absolute_pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.absolute_pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        if not self.mask_center:
            # get center mask start index & ending index
            h=torch.sqrt(torch.tensor(L)) #same as w
            start = int(torch.round((self.img_size-self.num_low_freqs)/2/self.img_size*h))
            end = int(torch.round((self.img_size+self.num_low_freqs)/2/self.img_size*h))

            # get overall center mask index to preserve
            ids_low_freq = torch.tensor([i+h*j for i in range(start, end) for j in range(start, end)], device=self.absolute_pos_embed.device) #[l]
            # order randomization
            low_noise = torch.rand(len(ids_low_freq), device=self.absolute_pos_embed.device)
            low_ids_shuffle = torch.argsort(low_noise)
            ids_low_freq = torch.gather(ids_low_freq, dim=0, index=low_ids_shuffle)
            # only preserve 80% of center in default
            len_low_keep = int(len(ids_low_freq)*0.8)
            ids_low_keep = ids_low_freq[:len_low_keep]

            # concat center index to the front & delete duplicated index
            _ids_shuffle = torch.ones(ids_shuffle.shape, device=self.absolute_pos_embed.device)
            for i in range(len(ids_low_keep)):
                _ids_shuffle = _ids_shuffle==(ids_shuffle!=ids_low_keep[i]) # True : not included in ids_low_keep, accumulate the False 
            _ids_shuffle=_ids_shuffle.nonzero(as_tuple=True)[1].view(N, -1) # get True index (N,L-l)
            _ids_shuffle = torch.gather(ids_shuffle, dim=1, index=_ids_shuffle) # get elements not included in ids_low_keep
            ids_shuffle = torch.cat([ids_low_keep.repeat(N,1), _ids_shuffle], dim=1).type(torch.int64)   

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.absolute_pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #(N,L)

        return ids_keep, ids_restore, mask

    def forward_encoder(self, x, mask_ratio):
        ids_keep, ids_restore, mask = self.masking_id(x.size(0), mask_ratio)

        if self.hifeat:
            x = self.forward_features(x, ids_keep=ids_keep, return_hifeat=True)
            h,m,l = x
            B,N,_ = l.shape
            x = torch.cat([h.reshape(B,N,-1), m.reshape(B,N,-1), 1], dim=-1)
            x = self.norm(x)
        else:
            x = self.forward_features(x, ids_keep=ids_keep)

        return x, mask, ids_restore


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if self.train and self.mae:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #(1,1,D)->(N,L*0.75,D)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))

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

        return None, x

    def forward_loss(self, imgs, pred, mask, ssl_masks, full=None):
        """
        imgs: [N, 2, H, W]
        pred: [N, L, p*p*2]
        mask: [N, L], 0 is keep, 1 is remove, 
        ssl_masks: [N, 1, H, W] 0 is keep, 1 is remove
        sp_masks: [N, 1, H, W] 0 is remove, 1 is keep
        """
        N,L,_=pred.shape

        sp_masks = 1-ssl_masks
        sp_masks = self.patchify(sp_masks)
        if full is not None:
            target = self.patchify(full)
        else:
            target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # loss = (pred - target) ** 2
        loss = torch.abs(pred - target)
        if self.ssl: # calculate loss in only acquired data
            loss = loss*sp_masks
        if self.divide_loss is not None:
            divide_loss = self.patchify(self.divide_loss)
            loss = loss*divide_loss.to(loss.device)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / N  # mean loss on removed patches
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        if self.ssl:
            loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
        else:
            loss = loss.sum() / N  # mean loss on every patches

        return loss
    
    def forward_sp_loss(self, pred, full, mask, ssl_masks):
        """
        imgs: [N, 2, H, W]
        pred: [N, L, p*p*2]
        mask: [N, L], 0 is keep, 1 is remove, 
        ssl_masks: [N, 1, H, W] 0 is keep, 1 is remove
        calculate only predicted region.
        """
        N,L,_=pred.shape

        ssl_masks = self.patchify(ssl_masks)
        target = self.patchify(full)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = torch.abs(pred - target)

        loss = (loss*ssl_masks).mean(dim=-1)*(1-mask) + (loss.mean(dim=-1))*mask

        loss = loss.sum() / N  # mean loss on every patches

        return loss

    def forward_ssl_loss(self, pred1, pred2, mask1, mask2, ssl_masks):
        N,L,_ = pred1.shape
        ssl_masks = self.patchify(ssl_masks)

        sslloss = torch.abs(pred1-pred2)
        sslloss = sslloss*ssl_masks
        if self.divide_loss is not None:
            divide_loss = self.patchify(self.divide_loss)
            sslloss = sslloss*divide_loss.to(sslloss.device)
        sslloss = sslloss.mean(dim=-1)
        # cmask = mask1*mask2
        # sslloss = (sslloss*cmask).sum() / cmask.sum() #only calculate in common masks
        sslloss = sslloss.sum()/N

        return sslloss
    
    def forward_img_loss(self, predimg, fullimg):
        N,_,_,_=predimg.shape
        imgloss = torch.sum(torch.abs(predimg-fullimg))/N
        return imgloss


    def forward(self, imgs, ssl_masks, full, mask_ratio=0.75):
        latent, mask, ids_restore= self.forward_encoder(imgs, mask_ratio)
        cls_pred, pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        #dc layer
        predfreq = self.unpatchify(pred)
        predfreq = imgs + predfreq*ssl_masks 

        #ifft
        predimg, fullimg = rifft2(predfreq, full, permute=True)
        maxnum = torch.max(fullimg)
        minnum = torch.min(fullimg)
        predimg = (predimg-minnum)/(maxnum-minnum+1e-08)
        fullimg = (fullimg-minnum)/(maxnum-minnum+1e-08)

        if self.train:
            loss = self.forward_loss(imgs, pred, mask, ssl_masks, full=full) #mask: 0 is keep, 1 is remove
            imgloss = self.forward_img_loss(predimg, fullimg)
            return loss, imgloss, torch.tensor([0], device=loss.device), self.unpatchify(pred), mask
        else: #not train, not ssl
            return self.unpatchify(pred), mask



def mae_hivit_base_8_768(**kwargs):
    model = HiViTMaskedAutoencoder(
        in_chans=2, embed_dim=768, depths=[6], num_heads=12, stem_mlp_ratio=3., mlp_ratio=4., 
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=16, hifeat=False,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_hivit_small_4_768(**kwargs):
    model = HiViTMaskedAutoencoder(
        in_chans=2, embed_dim=768, depths=[1,1,4], num_heads=12, stem_mlp_ratio=3., mlp_ratio=4., 
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16, hifeat=False,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

    
def hivit_base_8_768(**kwargs):
    model = HiViTMaskedAutoencoder(
        in_chans=2, embed_dim=768, depths=[1,1,8], num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs)
    return model

def hivit_small_4_768(**kwargs):
    model = HiViTMaskedAutoencoder(
        in_chans=2, embed_dim=768, depths=[1,1,4], num_heads=12, stem_mlp_ratio=3., mlp_ratio=4., 
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16, hifeat=False,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs)
    return model


mae_hivit_base = mae_hivit_base_8_768
mae_hivit_small = mae_hivit_small_4_768

hivit_base = hivit_base_8_768
hivit_small = hivit_small_4_768