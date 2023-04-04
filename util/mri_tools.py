import torch
import torch.fft


def fftshift(x, axes=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def ifftshift(x, axes=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fft2(data):
    assert data.shape[-1] == 2

    #should i add this line?
    data = ifftshift(data, axes=(-3,-2))


    data = torch.view_as_complex(data)
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)
    data = fftshift(data, axes=(-3, -2))
    return data



def ifft2(data):
    assert data.shape[-1] == 2
    data = ifftshift(data, axes=(-3, -2))
    data = torch.view_as_complex(data)
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data)

    #should i add this line?
    data = fftshift(data, axes=(-3,-2))
    return data


def permuteforward(data): #[c,h,w] -> [h,w,c]
    assert data.shape[-3] in [1,2]
    if data.dim()==3:
        data = data.permute(1,2,0)
    elif data.dim()==4:
        data = data.permute(0,2,3,1)
    else:
        raise SyntaxError('tensor dimension should be 3 or 4')
    return data

def permuteback(data): #[h,w,c] -> [c,h,w]
    assert data.shape[-1] in [1,2]
    if data.dim()==3:
        data = data.permute(2,0,1)
    elif data.dim()==4:
        data = data.permute(0,3,1,2)
    else:
        raise SyntaxError('tensor dimension should be 3 or 4')
    return data

def rfft2(*data, permute=False):
    def _rfft2(dt):
        if permute:
            dt = permuteforward(dt)
        assert dt.shape[-1] == 1
        dt = torch.cat([dt, torch.zeros_like(dt)], dim=-1)
        dt = fft2(dt)
        if permute:
            dt = permuteback(dt)
        return dt
    if len(data)==1:
        return _rfft2(data[0])
    else:
        return [_rfft2(dt) for dt in data]


def rifft2(*data, permute=False):
    def _rifft2(dt):
        if permute:
            dt = permuteforward(dt)
        assert dt.shape[-1] == 2
        dt = ifft2(dt)
        dt = (dt**2).sum(dim=-1).sqrt().unsqueeze(-1)
        # dt = dt[..., 0].unsqueeze(-1)
        if permute:
            dt = permuteback(dt)
        return dt
    if len(data)==1:
        return _rifft2(data[0])
    else:
        return [_rifft2(dt) for dt in data]


def rA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    return data


def rAt(data, mask):
    assert data.shape[-1] == 2
    data = ifft2(data * mask)
    data = (data**2).sum(dim=-1).sqrt().unsqueeze(-1)
    # data = data[..., 0].unsqueeze(-1)
    return data


def rAtA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    data = ifft2(data)
    data = (data**2).sum(dim=-1).sqrt().unsqueeze(-1)
    # data = data[..., 0].unsqueeze(-1)
    return data


def normalize(arr, eps=1e-08): #[0,1] for imgspace
    max = torch.max(arr)
    min = torch.min(arr)
    arr = (arr-min)/(max-min+eps)
    return arr


def scale(arr): #[-6,6] for kspace
    absmax = torch.max(torch.abs(arr))
    arr = arr/absmax*10
    return arr