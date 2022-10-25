import os, glob
from turtle import forward
import torch
import random
import pickle
import numpy as np
from typing import Ontional, Sequence, Tuple, Union

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.mri_tools import rfft2

''' FMRIDataset
input: kspace
output: subsampled image

load kspace
-> to tensor (complex stack in channel)
-> subsample kspace
-> ifft to image
-> absolute & Root-Sum-of-Square if multicoil data
-> normalize(mean=0, std=1) * clamp(-6,6)
'''

''' FMRIDataset
input: kspace
output: subsampled kspace

load kspace
-> to tensor (complex stack in channel)
-> subsample kspace
(-> normalize) ???
'''


class FMRIDataset(Dataset):
    def __init__(self, opt, mode='train') -> None:
        self.opt = opt
        self.mode = mode

        #downsample
        self.down = opt.down
        self.down_factor = opt.downsample
        self.low_freq_ratio = opt.low_freq_ratio
        self.rng = np.random.RandomState(opt.seed)

        self.datalist = glob.glob(os.path.join(opt.data_path, opt.dataset, mode, '**', '.pkl'))

    def __getitem__(self, idx):
        with open(self.datalist[idx], 'rb') as f:
            kdata = pickle.load(f)
        kdata = np.array(kdata)
        kdata = self.to_tensor(kdata)
        if self.down_factor>1:
            down_kdata, mask, num_low_freqs = self.downsample(kdata, self.low_freq_ratio, self.down_factor)
            return {'down': down_kdata, 'full': kdata, 'mask': mask, 'num_low_freqs': num_low_freqs}
        else:
            return {'down': None, 'full': kdata, 'mask': None, 'num_low_freqs': None}


    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr):
            arr = np.stack((arr.real, arr.img))
        return torch.from_numpy(arr)
    

    def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
        return torch.view_as_complex(data).numpy()


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

        return downsampled_data, mask, num_low_freqs


    def reshape_mask(self, mask:np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape"""
        h = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = h #[1,h,1]
        
        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))