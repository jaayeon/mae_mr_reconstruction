# https://github.com/chunmeifeng/MTrans

import torch
from torch import nn
from einops.layers.torch import Rearrange
from .mca import CrossTransformer
from util.mri_tools import rifft2, rfft2, normalize
from functools import partial

class ReconstructionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReconstructionHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        return out


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, in_chans=2, embed_dim=768, patch_size=16, patch_direction='pe'):
        super().__init__()
        """
        imgs: (N, c, H, W) --> (N, H, cxW)
        x: (N, L, D)
        """
        self.img_size = img_size
        self.in_chans = in_chans
        self.pd = patch_direction
        if patch_direction=='2d':
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
            self.num_patches = (img_size//patch_size)**2
        else:
            self.proj = nn.Linear(in_chans*img_size, embed_dim)
            self.num_patches = img_size

        self.patch_size = [patch_size, patch_size]

    def forward(self, imgs):
        if self.pd=='ro':
            x = torch.einsum('nchw->nhwc', imgs)
            x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
            x = self.proj(x)
        elif self.pd=='pe':
            x = torch.einsum('nchw->nwhc', imgs)
            x = x.reshape(shape=(imgs.shape[0], self.img_size, self.img_size*self.in_chans))
            x = self.proj(x)
        elif self.pd=='2d':
            x = self.proj(imgs)
            x = x.flatten(2).transpose(1,2)
            
        return x


class CrossMaskedAutoencoderViT(nn.Module):
    def __init__(self, patch_direction=['pe','ro','2d'], domain='kspace', img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mae=True, ssl=False, mask_center=False, 
                 num_low_freqs=None, guided_attention=0., regularize_attnmap=False):
        super().__init__()

        #self.head = ReconstructionHead(in_chans, 32)
        self.patch_embed1 = PatchEmbed(img_size, in_chans, embed_dim, patch_size, patch_direction=patch_direction[0])
        self.patch_embed2 = PatchEmbed(img_size, in_chans, embed_dim, patch_size, patch_direction=patch_direction[1])

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # same for encoder, decoder
        self.pos_embed1 =  nn.Parameter(torch.randn(1, self.patch_embed1.num_patches+1, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.randn(1, self.patch_embed2.num_patches+1, embed_dim))

        # encoder
        self.blocks = CrossTransformer(embed_dim, embed_dim, depth, num_heads, mlp_ratio)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        # decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = CrossTransformer(decoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio)
        self.decoder_norm1 = norm_layer(decoder_embed_dim)
        self.decoder_norm2 = norm_layer(decoder_embed_dim)
        self.decoder_pred1 = nn.Linear(decoder_embed_dim, patch_size**2*in_chans, bias=True)
        self.decoder_pred2 = nn.Linear(decoder_embed_dim, img_size*in_chans, bias=True)

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

    def initialize_weights(self):

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token1, std=.02)
        torch.nn.init.normal_(self.cls_token2, std=.02)
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
            x = torch.einsum('nchpwq->nhwpqc', x)

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
    

    def forward_encoder(self, x, mask_ratio, ssl_masks, given_ids_shuffle=None):
        # head
        #x = self.head(x)
        # embed patches
        x1 = self.patch_embed1(x)
        x2 = self.patch_embed2(x)

        # pos embedding
        x1 += self.pos_embed1[:,1:,:]
        x2 += self.pos_embed2[:,1:,:]

        # masking
        if self.train and self.mae:
            x1, mask1, ids_restore1, pair_ids1 = self.random_masking(x1, mask_ratio, ssl_masks, given_ids_shuffle)
            x2, mask2, ids_restore2, pair_ids2 = self.random_masking(x2, mask_ratio, ssl_masks, given_ids_shuffle)
        else: 
            mask1=mask2=None
            ids_restore1=ids_restore2=None
            pair_ids1=pair_ids2=None

        # append cls token
        cls_token1 = self.cls_token1 + self.pos_embed1[:,:1,:]
        cls_token2 = self.cls_token2 + self.pos_embed1[:,:1,:]
        cls_token1 = cls_token1.expand(x1.shape[0],-1,-1)
        cls_token2 = cls_token2.expand(x1.shape[0],-1,-1)
        x1 = torch.cat((cls_token1, x1), dim=1)
        x2 = torch.cat((cls_token2, x2), dim=1)

        # apply Encoder
        x1, x2 = self.blocks(x1, x2)
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        return [x1, mask1, ids_restore1], [x2, mask2, ids_restore2]
    
    def forward_decoder(self, set1, set2):
        
        x1, mask1, ids_restore1 = set1
        x2, mask2, ids_restore2 = set2

        if self.train and self.mae:
            mask_tokens1 = self.mask_token.repeat(x1.shape[0], ids_restore1.shape[1] + 1 - x1.shape[1], 1)
            x1_ = torch.cat([x1[:,1:,:], mask_tokens1], dim=1)
            x1_ = torch.gather(x1_, dim=1, index=ids_restore1.unsqueeze(-1).repeat(1,1,x1.shape[2]))
            x1 = torch.cat([x1[:,:1,:], x1_], dim=1)

            mask_tokens2 = self.mask_token.repeat(x2.shape[0], ids_restore2.shape[1] + 1 - x2.shape[1], 1)
            x2_ = torch.cat([x2[:,1:,:], mask_tokens2], dim=1)
            x2_ = torch.gather(x2_, dim=1, index=ids_restore2.unsqueeze(-1).repeat(1,1,x2.shape[2]))
            x2 = torch.cat([x2[:,:1,:], x2_], dim=1)

        # add pos embed
        x1 += self.pos_embed1
        x2 += self.pos_embed2

        if not torch.isfinite(x1).all() or not torch.isfinite(x2).all():
            print('anomaly detected: input of decoder')
        
        # apply Decoder
        x1, x2 = self.decoder_blocks(x1, x2)

        if not torch.isfinite(x1).all() or not torch.isfinite(x2).all():
            print('anomaly detected: output of decoder')
        
        x1 = self.decoder_norm1(x1)
        x2 = self.decoder_norm2(x2)

        x1 = self.decoder_pred1(x1)
        x2 = self.decoder_pred2(x2)

        return x1[:,1:,:], x2[:,1:,:]
    

    def forward_loss(self, imgs, pred, full=None, patch_direction='ro'):
        N,L,D = pred.shape
        if full is not None:
            target = self.patchify(full, patch_direction=patch_direction)
        else:
            target = self.patchify(imgs, patch_direction=patch_direction)

        loss = torch.abs(pred-target)
        loss = loss.mean(dim=-1).sum()/N

        return loss
    

    def forward_img_loss(self, predimg, fullimg):
        N,_,_,_ = predimg.shape
        imgloss = torch.sum(torch.abs(predimg-fullimg))/N
        return imgloss


    def forward(self, imgs, ssl_masks, full, mask_ratio=0.75):

        set1, set2 = self.forward_encoder(imgs, mask_ratio, ssl_masks)
        pred1, pred2 = self.forward_decoder(set1, set2)

        # dc layer
        fpred1 = self.unpatchify(pred1, patch_direction=self.patch_direction[0])
        fpred2 = self.unpatchify(pred2, patch_direction=self.patch_direction[1])

        fpred1 = imgs + fpred1*ssl_masks
        fpred2 = imgs + fpred2*ssl_masks

        #ifft
        ipred1, ipred2, ifull = rifft2(fpred1, fpred2, full, permute=True)
        maxnum = torch.max(ifull)
        minnum = torch.min(ifull)
        ipred1 = (ipred1-minnum)/(maxnum-minnum+1e-08)
        ipred2 = (ipred2-minnum)/(maxnum-minnum+1e-08)
        ifull = (ifull-minnum)/(maxnum-minnum+1e-08)

        if self.train and self.ssl:
            '''
            not implemented
            '''
            pass

        if self.train and self.ssl:
            '''
            not implemented
            '''
            pass
        elif self.train and not self.ssl:
            loss = self.forward_loss(imgs, pred1, full=full, patch_direction=self.patch_direction[0])
            #loss2 = self.forward_loss(imgs, pred2, full=full, patch_direction=self.patch_direction[1])
            #loss = (loss1+loss2)/2
            imgloss = self.forward_img_loss(ipred1, ifull)
            #imgloss2 = self.forward_img_loss(ipred2, ifull)
            #imgloss = (imgloss1+imgloss2)/2

            return loss, imgloss, torch.tensor([0],device=loss.device), torch.tensor([0],device=loss.device)
        else: # test
            # return (fpred1+fpred2)/2
            # return fpred2
            return fpred1
        

def mae_cross_small_4_768(**kwargs):
    model = CrossMaskedAutoencoderViT(
        embed_dim=768, depth=4, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def vit_alt_small_4_768(**kwargs):
    model = CrossMaskedAutoencoderViT(
        embed_dim=768, depth=2, num_heads=12,
        decoder_embed_dim=768, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), mae=False, **kwargs
    )
    return model

mae_cross_small = mae_cross_small_4_768
vit_cross_small = vit_alt_small_4_768