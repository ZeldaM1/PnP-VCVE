# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, negative_slope=0.1):
        super(QFAttention, self).__init__()

        self.res = nn.Sequential(*[
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,bias=bias)
        ])

    def forward(self,x, gamma, beta):
        res = (gamma)*self.res(x) + beta
        return x + res

