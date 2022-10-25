# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, ssl_loss=False, ssl_weight=1, no_center_mask=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.ssl_loss = ssl_loss
        self.ssl_weight = ssl_weight
        self.no_center_mask = no_center_mask
        self.train = True
        self.img_size = img_size

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
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
        imgs: (N, c, H, W)
        x: (N, L, patch_size**2 *c)
        """
        c=2
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, num_low_freqs):
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
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        if self.no_center_mask:
            # get center mask start index & ending index
            h=torch.sqrt(torch.tensor(L)) #same as w
            start = int(torch.round((self.img_size-num_low_freqs)/2/self.img_size*h))
            end = int(torch.round((self.img_size+num_low_freqs)/2/self.img_size*h))

            # get overall center mask index to preserve
            ids_low_freq = torch.tensor([i+h*j for i in range(start, end) for j in range(start, end)], device=x.device) #[l]

            # concat center index to the front & delete duplicated index
            _ids_shuffle = torch.ones(ids_shuffle.shape, device=x.device)
            for i in range(len(ids_low_freq)):
                _ids_shuffle = _ids_shuffle==(ids_shuffle!=ids_low_freq[i]) # True : not included in ids_low_freq
            _ids_shuffle=_ids_shuffle.nonzero(as_tuple=True)[1].view(N, -1) # get True index (N,L-l)
            _ids_shuffle = torch.gather(ids_shuffle, dim=1, index=_ids_shuffle) # get elements not included in ids_low_freq
            ids_shuffle = torch.cat([ids_low_freq.repeat(N,1), _ids_shuffle], dim=1).type(torch.int64)

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(N,L*0.75's index,D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #(N,L)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, num_low_freqs):

        #double the image num for ssl loss
        if self.ssl_loss :
            x = torch.cat((x,x),0) #N,C,H,W -> 2*N,C,H,W (batch index=[0,1,2,3,...,N-1,0,1,2,3,...,N-1])
            '''
            index_ = torch.cat((torch.arange(0,x.shape[0]), torch.arange(0,x.shape[0]))) 
            _, index_ = torch.sort(index_) #0, N, 1, N+1, 2, N+2, ...
            x = torch.gather(x, dim=0, index=index_.view(-1,1,1,1).repeat(1,1,x.shape[-2],x.shape[-1])) #[0,1,2,...,N-1,0,1,2,3,...,N-1] -> [0,0,1,1,2,2,...,N-1,N-1]
            '''

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if self.train:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio, num_low_freqs)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.train:
            return x, mask, ids_restore
        else:
            return x, None, None


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        if self.train:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #(1,1,D)->(N,L*0.75,D)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> to add positional embed
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, ssl_masks):
        """
        imgs: [N, 2, H, W]
        pred: [N, L, p*p*2]
        mask: [N, L], 0 is keep, 1 is remove, 
        ssl_masks: [N, 1, H, W] 0 is keep, 1 is remove
        sp_masks: [N, 1, H, W] 0 is remove, 1 is keep
        """
        N,L,_=pred.shape
        if self.ssl_loss:
            imgs = torch.cat((imgs,imgs),0)
            ssl_masks = torch.cat((ssl_masks, ssl_masks),0)

        sp_masks = torch.zeros(ssl_masks.shape, device=imgs.device)
        sp_masks[ssl_masks==0]=1

        ssl_masks = self.patchify(ssl_masks)
        sp_masks = self.patchify(sp_masks)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # loss = (pred - target) ** 2
        loss = torch.abs(pred - target)
        loss = loss*sp_masks
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / N  # mean loss on removed patches
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # loss = loss.sum() / N # mean loss on every patches

        # if not torch.isfinite(loss):
        #     print("sploss is {}, stopping training".format(loss))

        if self.ssl_loss:
            sslloss = self.get_ssl_loss(pred, ssl_masks)
            loss = loss+self.ssl_weight*sslloss
            # if not torch.isfinite(sslloss):
            #     print('sslloss is {}, stopping training'.format(sslloss))
        return loss
    
    def get_ssl_loss(self, pred, ssl_masks):
        #self-supervised loss for missing ground-truth pixels
        #to maximize the same semantic information of each input with different mask augmentations
        N,L,_ = pred.shape
        sslloss=0
        for i in range(int(N/2)):
            pred1 = pred[i,:,:]
            pred2 = pred[i+int(N/2),:,:]
            mask = ssl_masks[i,:,:]
            # loss = (pred1-pred2)**2
            loss = torch.abs(pred1-pred2)

            loss = (loss*mask).sum()/N/L*2
            # loss = loss.sum() #mean loss on every piexles

            sslloss+=loss
        return sslloss

    def forward(self, imgs, ssl_masks, mask_ratio=0.75, num_low_freqs=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, num_low_freqs)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        if self.train:
            loss = self.forward_loss(imgs, pred, mask, ssl_masks) #mask: 0 is keep, 1 is remove
            return loss, pred, mask
        else: 
            loss = None
            return loss, self.unpatchify(pred), mask


def mae_vit_base_patch16_uniform_dec768d12b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, in_chans=2, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_uniform = mae_vit_base_patch16_uniform_dec768d12b #decoder: 768 dim, 12 blocks