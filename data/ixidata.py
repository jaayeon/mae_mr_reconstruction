import os, glob
from turtle import forward
import torch
import random
import pickle
import numpy as np
from typing import Sequence

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.mri_tools import rfft2

''' IXIDataset
input: img
output: downsampled kspace

load img
-> to tensor
-> normalize [0,1]
-> fft to kspace
-> scale [-1,1]
-> subsample kspace
'''


class IXIDataset(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode

        #downsample
        self.down = opt.down
        self.down_factor = opt.downsample
        self.low_freq_ratio = opt.low_freq_ratio
        self.rng = np.random.RandomState(opt.seed)

        self.datalist = glob.glob(os.path.join(opt.data_path,opt.dataset,mode,'*','*','*.pkl'))
        self.transform = transforms.ToTensor()


    def __getitem__(self, idx):
        with open(self.datalist[idx], 'rb') as f:
            imgdata = pickle.load(f)
        imgdata = torch.tensor(imgdata).unsqueeze(-1)
        imgdata = self.normalize(imgdata)
        kdata = rfft2(imgdata)
        kdata = self.scale(kdata)
        kdata = kdata.permute(2,0,1)

        if self.down_factor>1:
            down_kdata, mask, num_low_freqs = self.downsample(kdata, self.low_freq_ratio, self.down_factor) #mask shape: [1,h,1]
            mask2d = self.mk_2d_mask(mask=mask, shape=kdata.shape)
            return {'down': down_kdata, 'full': kdata, 'mask': mask2d, 'num_low_freqs': num_low_freqs}
        else:
            mask2d = self.mk_2d_mask(shape=kdata.shape)
            return {'down': kdata, 'full': kdata, 'mask': mask2d, 'num_low_freqs': None}


    def normalize(self, arr, eps=1e-08): #[0,1] for spatial domain
        max = torch.max(arr)
        min = torch.min(arr)
        arr = (arr-min)/(max-min+eps)
        return arr
    

    def scale(self, arr): #[-1~1] for kspace
        absmax = torch.max(torch.abs(arr))
        arr = arr/absmax
        return arr


    def mk_2d_mask(self, mask=None, shape=(2,256,256)):
        c,h,w=shape
        #mask: 0 is remove, 1 is keep -> shuffle to 0 is keep, 1 is remove
        mask_ = torch.zeros(mask.shape, device=mask.device)
        mask_[mask==0]=1
        if mask is not None:
            mask2d=torch.ones(c,h,w)*mask_
        else:
            mask2d = torch.zeros(c,h,w)
        return mask2d


    def downsample(self, arr, low_freq_ratio: float, acceleration: int):
        c,h,w=arr.shape
        num_low_freqs = int(h/acceleration*low_freq_ratio) #select 70% from center line
        num_high_freqs = int(h/acceleration)-num_low_freqs

        #center_mask
        center_mask = np.zeros(h, dtype=np.float32)
        pad = (h - num_low_freqs + 1) // 2
        center_mask[pad : pad+num_low_freqs]=1
        assert center_mask.sum() == num_low_freqs
        center_mask = self.reshape_mask(center_mask, arr.shape)

        #acceleration mask
        if self.down=='random': 
            """
            #have to fix for each data


            
            """
            prob = num_high_freqs/(h-num_low_freqs)
            accel_mask = self.rng.uniform(size=h)<prob
            accel_mask = self.reshape_mask(accel_mask, arr.shape)
        elif self.down=='uniform':
            adjusted_accel = int((h-num_low_freqs)/(num_high_freqs))
            accel_mask = np.zeros(h, dtype=np.float32)
            accel_mask[0::adjusted_accel]=1
            accel_mask = self.reshape_mask(accel_mask, arr.shape)
            
        #apply mask
        mask = torch.max(center_mask, accel_mask)
        downsampled_data = arr*mask+0.0

        return downsampled_data, mask, torch.tensor(num_low_freqs)


    def reshape_mask(self, mask:np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape"""
        h = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = h #[1,h,1]
        
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
    
    def __len__(self):
        return len(self.datalist)

    '''
    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr):
            arr = np.stack((arr.real, arr.img))
        return torch.from_numpy(arr)
    

    def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
        return torch.view_as_complex(data).numpy()
    '''