from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block #, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed, focal_gaussian
from util.mri_tools import rifft2, rfft2, normalize


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


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]
    

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, in_chans=2, embed_dim=768, patch_size=16):
        super().__init__()
        """
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        self.img_size = img_size
        self.in_chans = in_chans
        self.proj1d = nn.Linear(in_chans*img_size, embed_dim)
        self.proj2d = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_patches = img_size

    def forward(self, imgs, patch_direction='ro'):
        if patch_direction=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
            x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
            x = self.proj1d(x)
        elif patch_direction=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
            x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
            x = self.proj1d(x)
        elif patch_direction=='2d':
            x = self.proj2d(x)
            x = x.flatten(2).transpose(1,2)
            
        return x
        
        
class AltMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_direction=['pe','ro','2d'], domain='kspace', img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mae=True, ssl=False, mask_center=False, 
                 num_low_freqs=None, guided_attention=0., regularize_attnmap=False):
        super().__init__()
        # patch_direction: ['pe','ro'], ['pe','ro','2d'], ['pe','2d']

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, in_chans, embed_dim, patch_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        #Block.attn.forward = forward_wrapper(Block.attn)

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

        self.wrap_blocks()

        self.ssl = ssl
        self.mae = mae
        self.mask_center = mask_center
        self.train = True
        self.img_size = img_size
        self.num_low_freqs = num_low_freqs
        self.domain = domain
        self.in_chans = in_chans
        self.patch_direction = patch_direction

        self.depth = depth
        self.decoder_depth = decoder_depth
        self.guided_attention = guided_attention
        self.regularize_attnmap = True if regularize_attnmap else False

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
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w1 = self.patch_embed.proj1d.weight.data
        w2 = self.patch_embed.proj2d.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

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


    def patchify(self, imgs, patch_direction='ro'):
        """
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        if patch_direction=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
        elif patch_direction=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
        elif patch_direction=='2d':
            p = self.patch_embed.patch_size[0]
            h=w=imgs.shape[2]//p
            x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', imgs)

        x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
        return x


    def unpatchify(self, x, patch_direction='ro'):
        """
        x: (N, H, cxW)
        imgs: (N, c, H, W)
        """      
        if patch_direction=='ro':
            x = x.reshape(shape=(x.shape[0], self.img_size, self.img_size, self.in_chans))
            imgs = torch.einsum('nhwc->nchw', x)
        elif patch_direction=='pe':
            x = x.reshape(shape=(x.shape[0], self.img_size, self.img_size, self.in_chans))
            imgs = torch.einsum('nwhc->nchw', x)
        elif patch_direction=='2d':
            p = self.patch_embed.patch_size[0]
            h=w=int(x.shape[1]**.5)
            assert h*w == x.shape[1]
            x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], self.in_chans, h*p, w*p))
        
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
        else:
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

            # if mask_center==False
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

            #make pair
            """ 
            flip_ids_shuffle = torch.flip(ids_shuffle, dims=[1])
            low_noise = torch.rand(len(ids_low_freq), device=x.device)
            low_ids_shuffle = torch.argsort(low_noise)
            ids_low_freq = torch.gather(ids_low_freq, dim=0, index=low_ids_shuffle)
            new_ids_low_keep = ids_low_freq[:len_low_keep]

            _flip_ids_shuffle = torch.ones(flip_ids_shuffle.shape, device=x.device)
            for i in range(len_low_keep):
                _flip_ids_shuffle = _flip_ids_shuffle==(flip_ids_shuffle!=new_ids_low_keep[i])
            _flip_ids_shuffle=_flip_ids_shuffle.nonzero(as_tuple=True)[1].view(N, -1)
            _flip_ids_shuffle = torch.gather(flip_ids_shuffle, dim=1, index=_flip_ids_shuffle)
            flip_ids_shuffle = torch.cat([new_ids_low_keep.repeat(N,1), _flip_ids_shuffle], dim=1).type(torch.int64)
            """


        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #(N, L*0.25)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #(N,L*0.75's index,D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #(N,L)

        return x_masked, mask, ids_restore, ids_shuffle


    def forward_encoder(self, x, mask_ratio, ssl_masks, given_ids_shuffle=None, patch_direction='pe'):
        # embed patches
        x = self.patch_embed(x, patch_direction=patch_direction)

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

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    
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
        '''
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        '''

        # loss = (pred - target) ** 2
        loss = torch.abs(pred - target)
        if self.ssl: # calculate loss in only acquired data 
            loss = loss*sp_masks
        '''
        if self.divide_loss is not None:
            divide_loss = self.patchify(self.divide_loss)
            loss = loss*divide_loss.to(loss.device)
        '''
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if self.ssl:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            loss = loss.sum() / N  # mean loss on every patches
        
        return loss

    def forward_kspace_loss(self, down, full):
        N,_,_,_=down.shape
        kspaceloss = torch.sum(torch.abs(down-full))/N
        return kspaceloss
        
    def forward_img_loss(self, predimg, fullimg):
        N,_,_,_=predimg.shape
        imgloss = torch.sum(torch.abs(predimg-fullimg))/N
        return imgloss

    def forward(self, imgs, ssl_masks, full, mask_ratio=0.75):

        for pd in self.patch_direction:
            latent1, mask1, ids_restore1, ids_shuffle = self.forward_encoder(imgs, mask_ratio, ssl_masks, patch_direction=pd)
            pred1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
            #dc layer
            predfreq1 = self.unpatchify(pred1)
            predfreq1 = imgs + predfreq1*ssl_masks
        
        # attention map regularization
        if self.train and self.regularize_attnmap:
                reg = self.attmap_regularization()
        else:
            reg = 0.0

        # ifft
        predimg1, fullimg = rifft2(predfreq1, full, permute=True)
        maxnum = torch.max(fullimg)
        minnum = torch.min(fullimg)
        predimg1 = (predimg1-minnum)/(maxnum-minnum+1e-08)
        fullimg = (fullimg-minnum)/(maxnum-minnum+1e-08)
        #predimg1 = normalize(predimg1)
        #fullimg = normalize(fullimg)

        if self.train and self.ssl: #for train w/ ssl
            imgs2 = imgs.clone()
            latent2, mask2, ids_restore2, _ = self.forward_encoder(imgs2, mask_ratio, ssl_masks)
            pred2 = self.forward_decoder(latent2, ids_restore2)
            #pred2 = pred2-init
            ppred2 = self.predictor(pred2)
                
            '''elif not self.train and self.ssl: #for test, use pair ids
            imgs2 = imgs.clone()
            latent2, mask2, ids_restore2, _ = self.forward_encoder(imgs2, mask_ratio, given_ids_shuffle=pair_ids)
            pred2 = self.forward_decoder(latent2, ids_restore2)
            ppred2 = self.predictor(pred2)
            '''
        else: #no ssl
            pass

        if self.train and self.ssl:
            loss1 = self.forward_loss(imgs, pred1, mask1, ssl_masks) #mask: 0 is keep, 1 is remove
            loss2 = self.forward_loss(imgs, pred2, mask2, ssl_masks) #mask: 0 is keep, 1 is remove
            sslloss1 = self.forward_ssl_loss(ppred1, pred2.detach(), mask1, mask2, ssl_masks)
            sslloss2 = self.forward_ssl_loss(pred1.detach(), ppred2, mask1, mask2, ssl_masks)
            return loss1+loss2, sslloss1+sslloss2, self.unpatchify(pred1), mask1, reg
        elif self.train and not self.ssl:
            loss = self.forward_loss(imgs, pred1, mask1, ssl_masks, full=full) #mask: 0 is keep, 1 is remove
            imgloss = self.forward_img_loss(predimg1, fullimg)

            return loss, imgloss, torch.tensor([0], device=loss.device), predfreq1, mask1, reg
        else: #not train, not ssl
            return predfreq1, mask1


def mae_alt_small_4_768(**kwargs):
    model = AltMaskedAutoencoderViT(
        embed_dim=1024, depth=8, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def vit_alt_small_4_768(**kwargs):
    model = AltMaskedAutoencoderViT(
        embed_dim=768, depth=4, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs
    )
    return model


mae_alt_small = mae_alt_small_4_768
vit_alt_small = vit_alt_small_4_768