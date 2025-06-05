# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from .basicvsr_net import ResidualBlocksWithInputConv,ResidualBlocksWithInputConvDynamic
from mmedit.models.common import flow_warp
from mmedit.models.registry import BACKBONES
from .iconvsr import IconVSR_restore_wo_refill_mv
 
from .iconvsr_mv import FVCDeformableAlignment,BasiceformableAlignment,VOSAlignment
 
@BACKBONES.register_module()
class IconVSR_restore_wo_refill_mv_ipb(IconVSR_restore_wo_refill_mv):
    def __init__(self,  mid_channels=64, num_blocks=30, with_cat=False, deform='vos',max_residue_magnitude=10, flow_inter='bilinear', **kwargs):
        super().__init__(mid_channels=mid_channels, num_blocks=num_blocks, flow_inter=flow_inter,**kwargs)
        self.with_cat=with_cat
        if deform=='basic':
            self.deform_align = BasiceformableAlignment(mid_channels, mid_channels, 3, padding=1, flow_inter=flow_inter, deform_groups=16, max_residue_magnitude=max_residue_magnitude)
        elif deform=='fvc':
            self.deform_align = FVCDeformableAlignment(mid_channels, mid_channels, 3, padding=1, deform_groups=16, max_residue_magnitude=max_residue_magnitude)
        elif deform=='vos':
            self.deform_align = VOSAlignment(flow_inter=flow_inter)
        elif deform=='stdf':
            raise TypeError('Not implemented yet')
        else:
            raise TypeError('Not such DCN type')

        self.backward_resblocks = ResidualBlocksWithInputConv(2 * mid_channels+3, mid_channels, num_blocks) if self.with_cat else ResidualBlocksWithInputConv(mid_channels+3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(3 * mid_channels+3, mid_channels, num_blocks) if self.with_cat else ResidualBlocksWithInputConv(2 * mid_channels+3, mid_channels, num_blocks) 

    def compute_flow(self, mvs):
        n, t, c, h, w = mvs.size()
        flows_forward_single = mvs[:, 1:, :2, ...] # IPBBBP
        flows_backward_single = mvs[:, :t-1, 2:, ...] # IPBBBP
 
        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
            flow_zero=torch.zeros((n, 1, 2, h, w),device=flows_forward_single.device)
            flows_backward = torch.cat([flows_backward_single, flow_zero, flows_forward_single.flip(dims=[1])], dim=1)
        else:
            flows_forward=flows_forward_single
            flows_backward=flows_backward_single

        return flows_forward, flows_backward

    def forward(self, lrs, QPs=None, slices=None, mvs=None):
        n, t, c, h_input, w_input = lrs.size()
        assert h_input >= 64 and w_input >= 64, (f'The height and width of inputs should be at least 64, but got {h_input} and {w_input}.')
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)
        lrs = self.spatial_padding(lrs)
        h, w = lrs.size(3), lrs.size(4)

        # compute optical flow and compute features for information-refill
        flows_forward, flows_backward = self.compute_flow(mvs)
        
        keyframe = (slices[:,:,0,0,0]==73) + (slices[:,:,0,0,0]==80)
        keyframe[:,-1]=1 # the first and the last must be keyframes
        keyframe[:,0]=1
 
        outputs_all = []
        for batch_idx in range(n):
            keyframe_batch=keyframe[batch_idx]
            outputs =[None for x in range(t)]
            outputs_forward=[]

            feat_prop = lrs.new_zeros(1, self.mid_channels, h, w)
            key_warp = lrs.new_zeros(1, self.mid_channels, h, w)#----------------TODO
            for i in range(t - 1, -1, -1):
                lr_curr = lrs[batch_idx, i, :, :, :].unsqueeze(0)
                if i < t - 1:  # no warping for the last timestep
                    flow = flows_backward[batch_idx, i, :, :, :].unsqueeze(0)
                    # warp the I/P frame
                    key_idx =  i+1+ int(torch.where(keyframe_batch[i+1:]==1)[0][0])
                    key_warp = self.deform_align(outputs[key_idx], flow)
                feat_prop = torch.cat([lr_curr, key_warp, feat_prop], dim=1) if self.with_cat else torch.cat([lr_curr, key_warp], dim=1)
                feat_prop = self.backward_resblocks(feat_prop)
                outputs[i]=feat_prop


            # forward-time propagation and upsampling
            feat_prop = torch.zeros_like(feat_prop)
            key_warp = torch.zeros_like(key_warp)#----------------TODO
            for i in range(0, t):
                lr_curr = lrs[batch_idx, i, :, :, :].unsqueeze(0)
                if i > 0:  # no warping for the first timestep
                    if flows_forward is not None:
                        flow = flows_forward[batch_idx, i - 1, :, :, :].unsqueeze(0)
                    else:
                        flow = flows_backward[batch_idx, -i, :, :, :].unsqueeze(0)
                    key_idx =  int(torch.where(keyframe_batch[0:i]==1)[0][-1])
                    key_warp = self.deform_align(outputs[key_idx], flow)
                feat_prop = torch.cat([lr_curr, key_warp, feat_prop, outputs[i]], dim=1) if self.with_cat else torch.cat([lr_curr, key_warp, outputs[i]], dim=1)
                feat_prop = self.forward_resblocks(feat_prop)
                outputs[i]=feat_prop

                out = self.lrelu(self.conv_hr(feat_prop))
                out = self.conv_last(out)
                out += lr_curr
                outputs_forward.append(out.squeeze(0))
            outputs_all.append(torch.stack(outputs_forward, dim=0))

        return torch.stack(outputs_all, dim=0) 

