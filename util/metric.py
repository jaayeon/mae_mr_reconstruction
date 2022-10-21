import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
#import ignite.metrics.FID as FID



#(1,c,h,w)... different resultf from (b,c,h,w)?
def calc_metrics(x, out, target, data_range=1, window_size=11, channel=1, size_average=True):

    #normalize
    x, out, target = normalize(x, out, target)
    
    #compute nmse
    noise_nmse = compute_NMSE(x, target)
    out_nmse = compute_NMSE(out, target)

    #compute psnr
    noise_loss, noise_psnr = compute_LOSS_PSNR(x, target)
    out_loss, out_psnr = compute_LOSS_PSNR(out, target)

    b,c,h,w=out.shape
    out=out.view(-1,1,h,w)
    target = target.view(-1,1,h,w)

    #compute ssim
    noise_ssim = compute_SSIM(x, target, data_range=data_range, window_size=window_size, channel=channel, size_average=size_average)
    out_ssim = compute_SSIM(out, target, data_range=data_range, window_size=window_size, channel=channel, size_average=size_average)
    

    results = {'noise_loss':noise_loss, 'loss':out_loss, 'noise_psnr':noise_psnr, 'psnr':out_psnr, 'noise_ssim':noise_ssim, 
                'ssim':out_ssim, 'noise_nmse':noise_nmse, 'nmse':out_nmse, }

    return results

def normalize(*img, eps=1e-08):
    def _normalize(img, eps=eps):
        max = torch.max(img)
        min = torch.min(img)
        img = (img-min)/(max-min+eps)
        return img
    if len(img)==1:
        return _normalize(img[0])
    else:
        return [_normalize(im) for im in img]
    

def compute_NMSE(pred, gt):
    return torch.linalg.norm(gt-pred)**2/torch.linalg.norm(gt)**2


def compute_LOSS_PSNR(img1, img2):
    mse_criterion = torch.nn.MSELoss()
    loss = mse_criterion(img1, img2)
    psnr = 10 * torch.log10(1 / loss)

    return loss, psnr


def compute_SSIM(img1, img2, data_range=1, window_size=11, channel=1, size_average=True):
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

