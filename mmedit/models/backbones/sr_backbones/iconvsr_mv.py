# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmedit.models.common import flow_warp

from mmcv.cnn import constant_init

class VOSAlignment(nn.Module):
    def __init__(self,  flow_inter='bilinear'):
        super().__init__()
        self.flow_inter=flow_inter
    
    def forward(self, feat_prop, flow):
        return flow_warp(feat_prop, flow.permute(0, 2, 3, 1),interpolation=self.flow_inter)
 
 
class FVCDeformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(FVCDeformableAlignment, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.deform_groups * self.kernel_size[0] * self.kernel_size[1]  * 3, 3, 1, 1)
        )

    def forward(self, ref_unwarped, offset_info):
        # deformable warp
        extra_feat = torch.cat([ref_unwarped, offset_info], dim=1)
        offset_and_mask = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(offset_and_mask, 3, dim=1)
        offset_map = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(ref_unwarped, offset_map, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

        # # motion compensation
        # deformed_f_ref = self.dcn_warp(offset_info, f_ref)
        # f_mc = torch.cat((deformed_f_ref, f_ref), dim=1)
        # f_mc = self.motion_compensate(f_mc)
        # f_bar = deformed_f_ref + f_mc
        # # feature space residual
        # res = f_cur - f_bar
         

class BasiceformableAlignment(ModulatedDeformConv2d):
    def __init__(self, *args,flow_inter=None, **kwargs):#
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(BasiceformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.deform_groups * self.kernel_size[0] * self.kernel_size[1]  * 3, 3, 1, 1)
        )
        self.init_offset()
        self.flow_inter=flow_inter

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, ref_unwarped, flow_1):
        ref_warped = flow_warp(ref_unwarped, flow_1.permute(0, 2, 3, 1),interpolation=self.flow_inter)
        extra_feat = torch.cat([ref_warped, flow_1], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_map = torch.cat((o1, o2), dim=1)       
        offset_map = offset_map + flow_1.flip(1).repeat(1, offset_map.size(1) // 2, 1, 1)
        # mask
        mask = torch.sigmoid(mask)
    # (torch.Size([2, 64, 64, 64]), torch.Size([2, 288, 64, 64]), torch.Size([2, 144, 64, 64]))
        return modulated_deform_conv2d(ref_unwarped, offset_map, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

 
