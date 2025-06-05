# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from .basicvsr_net import ResidualBlocksWithInputConvDynamic_drt
from mmedit.models.common import flow_warp,PixelShufflePack
from mmedit.models.registry import BACKBONES
from .domain_aware import Base_Predictor,Bias_Predictor,SEModule
 
from .iconvsr_ipb import IconVSR_restore_wo_refill_mv_ipb


 
@BACKBONES.register_module()
class IconVSR_restore_wo_refill_mv_ipb_fast_domain_dynamic_with_par(IconVSR_restore_wo_refill_mv_ipb):
    def __init__(self,  mid_channels=64, num_blocks=30, num_experts=10, num_group=1, expert_softmax=False, use_base_qp=False, with_bias=False, with_se=False, with_par=False, init_weight=False, one_layer=False, small_sft=False, blocktype='default',channel_first=False, drconv=False, sparse_val=False, vsr=False, align_key=False, **kwargs):
        super().__init__(mid_channels=mid_channels, num_blocks=num_blocks, **kwargs)

        self.use_base_qp=use_base_qp
        self.with_bias=with_bias
        self.with_par=with_par
        self.vsr=vsr
        self.align_key=align_key
        if with_bias:
            assert use_base_qp == True
            self.BiasePredictor = SEModule(mid_channels) if with_se else Bias_Predictor(mid_channels)  
        self.BasePredictor = Base_Predictor(nf=mid_channels, num_experts=num_experts, softmax=expert_softmax)  
        add_ch= 3
        forward_factor=2 if self.with_cat else 1
        backward_factor=3 if self.with_cat else 2
        self.backward_resblocks = ResidualBlocksWithInputConvDynamic_drt(forward_factor * mid_channels+add_ch, mid_channels, num_blocks, num_experts, with_bias, with_se=with_se, init_weight=init_weight, num_group=num_group, one_layer=one_layer, blocktype=blocktype, channel_first=channel_first,sparse_val=sparse_val)  
        self.forward_resblocks = ResidualBlocksWithInputConvDynamic_drt(backward_factor * mid_channels+add_ch, mid_channels, num_blocks, num_experts, with_bias, with_se=with_se, init_weight=init_weight, num_group=num_group, one_layer=one_layer, blocktype=blocktype, channel_first=channel_first,sparse_val=sparse_val)  
        
        if vsr: # upsample
            self.upsample1 = PixelShufflePack(
                mid_channels, mid_channels, 2, upsample_kernel=3)
            self.upsample2 = PixelShufflePack(
                mid_channels, 64, 2, upsample_kernel=3)
            self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)


    def forward(self, lrs, QPs=None, slices=None, mvs=None,base_QPs=None,par_map=None):
        used_QPs = base_QPs if self.use_base_qp else QPs 
        experts_weights = self.BasePredictor(used_QPs)
        if self.with_bias:
            gammas, betas = self.BiasePredictor(QPs)    

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

        outputs =[None for x in range(t)]
        outputs_forward=[]

        neighbor_warp = lrs.new_zeros(n, self.mid_channels, h, w)
        key_warp = lrs.new_zeros(n, self.mid_channels, h, w)#----------------TODO
        for i in range(t - 1, -1, -1):
            # lr_curr = lrs[:, i, :, :, :]
            # breakpoint()
            lr_curr = lrs[:, i, :, :, :] 
            if i < t - 1:  # no warping for the last timestep
                key_warp_list =[]
                neighbor_warp_list=[]
                for batch_idx in range(n):
                    flow = flows_backward[batch_idx, i, :, :, :].unsqueeze(0)
                    # warp the I/P frame
                    key_idx =  i+1+ int(torch.where(keyframe[batch_idx,i+1:]==1)[0][0])
                    key_fea=self.deform_align(outputs[key_idx][batch_idx].unsqueeze(0), flow)
                    key_warp_list.append(key_fea)
                    if self.align_key and (key_idx==i+1):
                        neighbor_warp_list.append(key_fea)
                    else:
                        neighbor_warp_list.append(outputs[i+1][batch_idx].unsqueeze(0))
                key_warp = torch.cat(key_warp_list,dim=0) 
                neighbor_warp = torch.cat(neighbor_warp_list,dim=0) 
            feat_prop = torch.cat([lr_curr, key_warp, neighbor_warp], dim=1) if self.with_cat else torch.cat([lr_curr, key_warp], dim=1)
            # (gamma, beta) =  (gammas[:, i, ...], betas[:, i, ...]) if self.with_bias else (None, None)
            if self.with_bias:
                gamma=gammas[:, i, ...]
                beta=betas[:, i, ...] if (betas is not None) else None
            else:
                (gamma, beta) = (None, None)
            
            inputs={'x': feat_prop, 'par':par_map[:, i, :, :, :], 'weights': experts_weights[:, i, ...],'gamma': gamma, 'beta': beta}
            feat_prop = self.backward_resblocks(inputs)['x']  
            outputs[i]=feat_prop

    # forward-time propagation and upsampling
        neighbor_warp = torch.zeros_like(feat_prop)
        key_warp = torch.zeros_like(key_warp) 
        for i in range(0, t):
            # lr_curr = lrs[:, i, :, :, :]
            lr_curr =  lrs[:, i, :, :, :] 
            if i > 0:  # no warping for the first timestep
                key_warp_list =[]
                neighbor_warp_list=[]
                for batch_idx in range(n):
                    if flows_forward is not None:
                        flow = flows_forward[batch_idx, i - 1, :, :, :].unsqueeze(0)
                    else:
                        flow = flows_backward[batch_idx, -i, :, :, :].unsqueeze(0)
                    key_idx =  int(torch.where(keyframe[batch_idx,0:i]==1)[0][-1])
                    key_fea = self.deform_align(outputs[key_idx][batch_idx].unsqueeze(0), flow)
                    key_warp_list.append(key_fea)
                    if self.align_key and (key_idx==i-1):
                        neighbor_warp_list.append(key_fea)
                    else:
                        neighbor_warp_list.append(outputs[i-1][batch_idx].unsqueeze(0))
                key_warp = torch.cat(key_warp_list,dim=0) 
                neighbor_warp = torch.cat(neighbor_warp_list,dim=0) 
            feat_prop = torch.cat([lr_curr, key_warp, neighbor_warp, outputs[i]], dim=1) if self.with_cat else torch.cat([lr_curr, key_warp, outputs[i]], dim=1)
            # (gamma, beta) =  (gammas[:, i, ...], betas[:, i, ...]) if self.with_bias else (None, None)
            if self.with_bias:
                gamma=gammas[:, i, ...]
                beta=betas[:, i, ...] if (betas is not None) else None
            else:
                (gamma, beta) = (None, None)
            inputs={'x': feat_prop, 'par':par_map[:, i, :, :, :], 'weights': experts_weights[:, i, ...],'gamma': gamma, 'beta': beta}
            feat_prop = self.forward_resblocks(inputs)['x']    
            outputs[i]=feat_prop
            if self.vsr:
                out = self.lrelu(self.upsample1(feat_prop))
                out = self.lrelu(self.upsample2(out))
                out = self.lrelu(self.conv_hr(out))
                out = self.conv_last(out)
                base = self.img_upsample(lr_curr)
                out += base
                outputs_forward.append(out)
            else:
                out = self.lrelu(self.conv_hr(feat_prop))
                out = self.conv_last(out)
                out += lrs[:, i, :, :, :]
                outputs_forward.append(out)

        return torch.stack(outputs_forward, dim=1) 
 
  