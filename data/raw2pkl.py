from email.mime import base
import pickle
import os, glob
import nibabel as nib
import numpy as np
import imageio

'''
T2 #579: normally (256,256,130), except IOP (256,256,136)
T1 #582: normally (256,256,150), except IOP (256,256,146)
PD #579: normally (256,256,130), except IOP (256,256,136)
'''

cutoff_end=0.85
cutoff_start=0.15

data_dir = '../../data/ixi/*/*/*.nii.gz' #256,256,130, except IOP: 256,256,136
datalist = glob.glob(data_dir)

for i,datapath in enumerate(datalist):
    basename = os.path.basename(datapath)
    basename = basename.split('.')[0]
    newdir = os.path.join('/'.join(datapath.split('/')[:-1]), basename)
    data = nib.load(datapath)
    arr = data.get_fdata()
    print('{}: '.format(datapath), arr.shape)
    arr = np.array(arr, dtype='float32')
    print('min: {}, max: {}'.format(np.min(arr), np.max(arr)))
    s = arr.shape[-1]
    ''' #checking
    os.mkdir('./tests/example{:02d}'.format(i))
    for si in range(s):
        imageio.imwrite('./tests/example{:02d}/{:03d}.tif'.format(i,si), arr[:,:,si])
    if i==20:
        exit()
    '''
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    for si in range(int(s*cutoff_start), int(s*cutoff_end)):
        newpath = os.path.join(newdir, '{}_{:03d}.pkl'.format(basename, si))
        with open(newpath, 'wb') as f:
            pickle.dump(arr[:,:,si], f)
    
