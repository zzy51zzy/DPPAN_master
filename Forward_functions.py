import numpy as np
import torch
import math
import torch.nn.functional as F

def FrFT_forward(obj, frft_order):
    propfield = frft2d(obj, frft_order)

    return propfield

def FrFT_backward(propfield, frft_order):
    obj = frft2d(propfield, -frft_order)

    return obj

def frft2d(x, a):

    # if x.dtype != torch.complex:
    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))


    ### x.size
    bsize, hsize, wsize = x.shape
    
    ### do frft for dim = -1
    x = frft_dim(x.reshape(bsize*hsize,wsize), a).reshape(bsize,hsize,wsize)

    ### do frft for dim = 0
    x = frft_dim(x.permute(0,2,1).reshape(bsize*wsize,hsize), a).reshape(bsize,hsize,wsize).permute(0,2,1)

    return x


def frft_dim(x, a, dim=-1):
  ### perform 1D a-order frft of x
    # x (Tensor: complex128) - the input tensor should be two-dimension tensor
    # dim (int, optional) - the dimension along which to take the one dimensional frft
    # a (Tensor: float64) - the fractional order of frft
    N = x.shape[dim]
    sN = math.sqrt(N)
    a = torch.fmod(a,4) # the fractional Fourier transform has a period of four
    # do special case
    if a == 0:
        return x
    elif a == 2:
        return torch.flip(x, dims=[dim])
    elif a == 1:
        return torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(x), dim=dim))/sN
    elif a == 3:
        return torch.fft.fftshift(torch.fft.ifft(torch.fft.fftshift(x), dim=dim))*sN   
    else:
        # reduce to interval 0.5 < a < 1.5
        if a>2:
            a = a - 2
            x = torch.flip(x, dims=[dim])
        if a>1.5:
            a = a - 1
            x = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(x), dim=dim))/sN
        if a<0.5:
            a = a + 1
            x = torch.fft.fftshift(torch.fft.ifft(torch.fft.fftshift(x), dim=dim))*sN       
        # the general case for 0.5 < a < 1.5
        alpha = a*torch.pi/2
        tana2 = torch.tan(alpha/2)
        sina = torch.sin(alpha)
        f = F.pad(interp_dim(x),(N-1,N-1), "constant", 0)
        # chirp premultiplication
        c_arg = (torch.pi/N/4*torch.arange(-2*N+2,2*N-1)**2).to(x.device)
        chrp_r = torch.cos(tana2*c_arg)
        chrp_i = -torch.sin(tana2*c_arg)
        chrp = torch.complex(chrp_r,chrp_i).unsqueeze(0)
        f = chrp*f
        # chirp convolution
        cc = (torch.pi/N/4)/sina
        c_arg2 = (torch.arange(-4*N+4,4*N-3)**2).to(x.device)
        ch_r = torch.cos(cc*c_arg2)
        ch_i = torch.sin(cc*c_arg2)
        ch = torch.complex(ch_r,ch_i).unsqueeze(0)
        Faf = fconv_dim(ch,f)
        Faf = Faf[:,4*N-4:8*N-7]*torch.sqrt(cc/torch.pi)
        Faf = chrp*Faf
        Faf = Faf[:,N-1:Faf.shape[dim]-N+1]
        norm_constant = torch.complex(torch.cos((1-a)*torch.pi/4), -torch.sin((1-a)*torch.pi/4))
        Faf = norm_constant*Faf[:,::2]
        return Faf

def interp_dim(x, dim=-1):
    # sinc interpolation
    N = x.shape[dim]
    y = torch.zeros_like(x)
    y = F.pad(y,(N-1,0), "constant", 0)
    # print(N)             # N=256
    # print(y.shape)       # (256,511)
    # print(x.shape)       # (256,256)
    y[:,::2] = x 
    xint = fconv_dim(y, torch.sinc(torch.arange(-2*N+3,2*N-2)/2).unsqueeze(0).to(x.device))
    xint = xint[:,2*N-3:xint.shape[dim]-2*N+3]
    return xint

def fconv_dim(x, y, dim=-1):
    # convolution by fft
    N = x.shape[dim] + y.shape[dim] - 1
    P = 2**math.ceil(math.log2(N))
    z = torch.fft.ifft(torch.fft.fft(x, P, dim=dim)*torch.fft.fft(y, P, dim=dim), dim=dim)
    z = z[:,0:N]
    return z

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, eta, beta):
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * eta.expand(size) + beta.expand(size)
