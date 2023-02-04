import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import scipy.stats as st

#
# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel array."""
#     interval = (2*nsig+1.)/(kernlen)
#     x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#     kernel = kernel_raw/kernel_raw.sum()
#     return kernel
#
#
# class GaussianBlur(nn.Module):
#     def __init__(self, kernel):
#         super(GaussianBlur, self).__init__()
#         self.kernel_size = len(kernel)
#         print('kernel size is {0}.'.format(self.kernel_size))
#         assert self.kernel_size % 2 == 1, 'kernel size must be odd.'
#
#         self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         self.weight = nn.Parameter(data=self.kernel, requires_grad=False)
#
#     def forward(self, x):
#         x1 = x[:,0,:,:].unsqueeze_(1)
#         x2 = x[:,1,:,:].unsqueeze_(1)
#         x3 = x[:,2,:,:].unsqueeze_(1)
#         padding = self.kernel_size // 2
#         x1 = F.conv2d(x1, self.weight, padding=padding)
#         x2 = F.conv2d(x2, self.weight, padding=padding)
#         x3 = F.conv2d(x3, self.weight, padding=padding)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x
#
#
# def get_gaussian_blur(kernel_size, device):
#     kernel = gkern(kernel_size, 2).astype(np.float32)
#     gaussian_blur = GaussianBlur(kernel)
#     return gaussian_blur.to(device)


import torch
import math
import torch.nn as nn


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

