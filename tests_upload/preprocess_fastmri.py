import h5py
import numpy as np
import pickle
import os, glob
import torch
import fastmri
from fastmri.data import transforms as T
import torchvision


'''
single-coil: k-space (# slices, 320, 320)

'''
data_train_dir = '../../data/fastmri/singlecoil_train_h5/*.h5'
data_train_list = glob.glob(data_train_dir)

if not os.path.exists('../../data/fastmri/train/img'):
    os.mkdir('../../data/fastmri/train/img')

for i, datapath in enumerate(data_train_list):
    print(datapath)
    hf = h5py.File(str(datapath))
    basename = os.path.basename(datapath).split('.')[0]
    print('Keys: ', list(hf.keys()))
    print('Attrs: ', dict(hf.attrs))
    kspace = hf['kspace'][()] # type: list
    img_stack =[]
    for j in range(10, len(kspace)-5):
        datawrite = os.path.join('/'.join(datapath.split('/')[:-2]), 'train', '{}_{:02d}.pkl'.format(basename, j))
        slice_kspace = T.to_tensor(kspace[j]) #convert from numpy array to pytorch tensor
        # slice_kspace = T.complex_center_crop(slice_kspace,(320,320))
        slice_kspace = fastmri.fft2c(T.complex_center_crop(fastmri.ifft2c(slice_kspace), (320,320))) #convert to image
        with open(datawrite, 'wb') as f:
            pickle.dump(slice_kspace, f)
            print('save {}'.format(datawrite))

        if i%10==0:
            slice_image = fastmri.ifft2c(slice_kspace)
            # print(torch.max(slice_kspace), torch.min(slice_kspace))
            slice_image_abs = fastmri.complex_abs(slice_image)
            h,w=slice_image_abs.shape
            # print(torch.max(slice_image_abs), torch.min(slice_image_abs))
            img_stack.append(slice_image_abs.reshape(1,1,h,w))

    if i%20==0:
        img_stack = torch.cat(img_stack, dim=0)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(img_stack, normalize=True, scale_each=True, nrow=6, padding=10, pad_value=1.),
            os.path.join('../../data/fastmri/train/img/{}.png'.format(basename))
        )
    


data_test_dir = '../../data/fastmri/singlecoil_test_h5/*.h5'
data_test_dir = glob.glob(data_test_dir)
if not os.path.exists('../../data/fastmri/test/img'):
    os.mkdir('../../data/fastmri/test/img')

for i, datapath in enumerate(data_train_list):
    print(datapath)
    hf = h5py.File(str(datapath))
    basename = os.path.basename(datapath).split('.')[0]
    print('Keys: ', list(hf.keys()))
    print('Attrs: ', dict(hf.attrs))
    kspace = hf['kspace'][()] # type: list
    img_stack =[]
    for j in range(10, len(kspace)-5):
        datawrite = os.path.join('/'.join(datapath.split('/')[:-2]), 'test', '{}_{:02d}.pkl'.format(basename, j))
        slice_kspace = T.to_tensor(kspace[j]) #convert from numpy array to pytorch tensor
        # slice_kspace = T.complex_center_crop(slice_kspace,(320,320))
        slice_kspace = fastmri.fft2c(T.complex_center_crop(fastmri.ifft2c(slice_kspace), (320,320))) #convert to image
        with open(datawrite, 'wb') as f:
            pickle.dump(slice_kspace, f)
            print('save {}'.format(datawrite))

        slice_image = fastmri.ifft2c(slice_kspace)
        # print(torch.max(slice_kspace), torch.min(slice_kspace))
        slice_image_abs = fastmri.complex_abs(slice_image)
        h,w=slice_image_abs.shape
        # print(torch.max(slice_image_abs), torch.min(slice_image_abs))
        img_stack.append(slice_image_abs.reshape(1,1,h,w))
        

    img_stack = torch.cat(img_stack, dim=0)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img_stack, normalize=True, scale_each=True, nrow=6, padding=10, pad_value=1.),
        os.path.join('../../data/fastmri/test/img/{}.png'.format(basename))
    )