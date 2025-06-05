 # Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,ResidualBlockNoBNDynamic,ResidualBlockNoBNDynamic_drt,ResidualBlockNoBNDynamic_drt_wo_qp, ResidualBlockNoBNDynamicSFT, ResidualBlockNoBNDynamic_cbam_conv, ResidualBlockNoBNDynamicSFT_res, ResidualBlockNoBNDynamic_cbam, flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from .domain_aware import Hsigmoid
class VOSAlignment(nn.Module):
    def __init__(self,  flow_inter='bilinear'):
        super().__init__()
        self.flow_inter=flow_inter
    
    def forward(self, feat_prop, flow):
        return flow_warp(feat_prop, flow.permute(0, 2, 3, 1),interpolation=self.flow_inter)
 

@BACKBONES.register_module()
class BasicVSRNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs, QPs=None, slices=None):
        """Forward function for BasicVSR.

        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)
        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')



@BACKBONES.register_module()
class MetabitNet(nn.Module):
    """BasicVSR network structure for video super-resolution.

    Support only x4 upsampling.
    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in each propagation branch.
            Default: 30.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self):

        super().__init__()

 
        mid_channels=64
        self.mid_channels=mid_channels
        self.use_base_qp=True
        # propagation branches
        # self.II_resblocks = ResidualBlocksWithInputConv(
        #     64 + 3, 64, 10)
        # self.IP_resblocks = ResidualBlocksWithInputConv(
        #     16 + 3, 16, 10)
        # self.IG_resblocks = ResidualBlocksWithInputConv(
        #     64, 64, 10)
        # self.PG_resblocks = ResidualBlocksWithInputConv(
        #     64, 64, 10)
        self.with_cat=True
        self.backward_resblocks = ResidualBlocksWithInputConv(64 , 64, 10)

        self.forward_resblocks = ResidualBlocksWithInputConv(64, 64, 10)

        # propagation branches
        self.deform_align = VOSAlignment(flow_inter='bilinear')



        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
    
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
   
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def spatial_padding(self, lrs):
        n, t, c, h, w = lrs.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        lrs = lrs.view(-1, c, h, w)
        lrs = F.pad(lrs, [0, pad_w, 0, pad_h], mode='reflect')

        return lrs.view(n, t, c, h + pad_h, w + pad_w)
        
    def compute_flow(self, mvs):
        n, t, c, h, w = mvs.size()
        flows_forward_single = mvs[:, 1:, :2, ...] # IPBBBP
        flows_backward_single = mvs[:, 1:, 2:, ...] # IPBBBP
 
        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
            flow_zero=torch.zeros((n, 1, 2, h, w),device=flows_forward_single.device)
            flows_backward = torch.cat([flows_backward_single, flow_zero, flows_backward_single.flip(dims=[1])], dim=1)
        else:
            flows_forward=flows_forward_single
            flows_backward=flows_backward_single

        return flows_forward, flows_backward

    def forward(self, lrs, QPs=None, slices=None, mvs=None,base_QPs=None,par_map=None):
        # lrs=torch.zeros([1, 5, 3, 180, 320]).to(lrs.device)
        # QPs=torch.zeros([1, 5, 1, 1, 1]).to(lrs.device)
        # slices=torch.zeros([1, 5, 1, 1, 1]).to(lrs.device)
        # mvs=torch.zeros([1, 5, 4,  180, 320]).to(lrs.device)
        # base_QPs=torch.zeros([1, 5, 1, 1, 1]).to(lrs.device)
        # par_map=torch.zeros([1, 5, 3,  180, 320]).to(lrs.device)
        used_QPs = base_QPs if self.use_base_qp else QPs 
      
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
        key_fea_zeros = lrs[0,0,:].new_zeros(1, self.mid_channels, h, w)
        
        #----------stage1: P to I  backward 
        for i in range(t - 1, -1, -1):
            # lr_curr = lrs[:, i, :, :, :]
            # breakpoint()
            lr_curr = lrs[:, i, :, :, :] 
            if i < t - 1:  # no warping for the last timestep
                key_warp_list =[]
                for batch_idx in range(n):
                    flow = flows_backward[batch_idx, i, :, :, :].unsqueeze(0)
                    # warp the I/P frame
                    find_B=torch.where(keyframe[batch_idx,i+1:]==0)[0]
                    if len(find_B)>0:
                        key_idx =  i+1+ int(find_B[0])  
                        key_fea=self.deform_align(outputs[key_idx][batch_idx].unsqueeze(0), flow)
                    else:
                        key_fea=key_fea_zeros
                    key_warp_list.append(key_fea)
                key_warp = torch.cat(key_warp_list,dim=0) 
            feat_prop = self.backward_resblocks(key_warp)  
            outputs[i]=feat_prop

    # forward-time propagation and upsampling
        neighbor_warp = torch.zeros_like(feat_prop)
        key_warp = torch.zeros_like(key_warp) 
        for i in range(0, t):
            # lr_curr = lrs[:, i, :, :, :]
            lr_curr =  lrs[:, i, :, :, :] 
            if i > 0:  # no warping for the first timestep
                key_warp_list =[]
                for batch_idx in range(n):
                    if flows_forward is not None:
                        flow = flows_forward[batch_idx, i - 1, :, :, :].unsqueeze(0)
                    else:
                        flow = flows_backward[batch_idx, -i, :, :, :].unsqueeze(0)
                    key_idx =  int(torch.where(keyframe[batch_idx,0:i]==1)[0][-1])
                    key_fea = self.deform_align(outputs[key_idx][batch_idx].unsqueeze(0), flow)
                    key_warp_list.append(key_fea)
                key_warp = torch.cat(key_warp_list,dim=0) 
            
            feat_prop = self.forward_resblocks(key_warp)   
            outputs[i]=feat_prop

            out = self.lrelu(self.conv_hr(feat_prop))
            out = self.conv_last(out)
            out += lrs[:, i, :, :, :]
            outputs_forward.append(out)

        return torch.stack(outputs_forward, dim=1) 
 
  

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)

class ResidualBlocksWithInputConvDynamic(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30, num_experts=10, with_bias=False, with_se=False, init_weight=False, gaintune=False, gainbias=False, num_group=1, one_layer=False, channel_first=False):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        self.input_conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])

        # residual blocks
        self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight, gaintune=gaintune, gainbias=gainbias, one_layer=one_layer,channel_first=channel_first)))

    def forward(self, feat):
        feat['x'] = self.input_conv(feat['x'])
        return self.main(feat)

def generate_indices(spa_mask, kernel_size):
    A = torch.arange(3).to(spa_mask.device).view(-1, 1, 1)
    # spa_mask=torch.ones_like(spa_mask)
    mask_indices = torch.nonzero(spa_mask.squeeze())
    

    # indices: dense to sparse (1x1)
    h_idx_1x1 = mask_indices[:, 0]
    w_idx_1x1 = mask_indices[:, 1]

    if kernel_size == 1:
        return h_idx_1x1, w_idx_1x1

    # indices: dense to sparse (3x3)
    mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A

    h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)
    w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)
 
    if kernel_size == 3:
        return h_idx_3x3, w_idx_3x3

class ResidualBlocksWithInputConvDynamic_drt(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30, num_experts=10, with_bias=False, with_se=False, init_weight=False, num_group=1, one_layer=False, blocktype='default',channel_first=True, sparse_val=False):
        super().__init__()
        self.blocktype=blocktype
        self.sparse_val=sparse_val
        # a convolution used to match the channels of the residual blocks
        self.input_conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        
        # residual blocks
        if blocktype=='drt':
        #     self.input_par = nn.Sequential(*[nn.Conv2d(4, 1, 1, 1, 0, bias=False), Hsigmoid()])
            # self.ave_pool=nn.AdaptiveAvgPool2d(1)
            # self.fc = nn.Sequential(
            #     nn.Linear(4, 4, bias=False),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4, 3, bias=False),
            # )
            # self.softmax=Hsigmoid() #nn.Softmax(dim=1)
                        # par=self.ave_pool(torch.cat([feat['x'][:,:3,...],feat['par']],dim=1))
            # b,c,h,w=par.size()
            # par = self.fc(par.view(b,c)).view(b,3,1,1,1)
            # par = self.softmax(par)#feat['par'] 

            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic_drt, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_se=with_se, init_weight=init_weight,  one_layer=one_layer,channel_first=channel_first, sparse_val=sparse_val)))
        elif blocktype=='drt_woqp':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic_drt_wo_qp, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_se=with_se, init_weight=init_weight,  one_layer=one_layer,channel_first=channel_first, sparse_val=sparse_val)))


    def forward(self, feat):
        if self.blocktype=='drt' or self.blocktype=='drt_woqp':
            # par = self.input_par(torch.cat([feat['x'][:,:3,...],feat['par']],dim=1))
            b,c,h,w=feat['par'].size()
            feat['par'] = feat['par'].view(b,c,1, h,w)
            if self.sparse_val and (not self.training):
                feat['sparse_0'] = generate_indices(feat['par'][:,0,...],1)
                feat['sparse_1'] = generate_indices(feat['par'][:,1,...],1)
                feat['sparse_2'] = generate_indices(feat['par'][:,2,...],1)
        feat['x'] = self.input_conv(feat['x'])

        
        
        return self.main(feat)

class ResidualBlocksWithInputConvDynamic_SFT(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30, num_experts=10, with_bias=False, with_se=False, init_weight=False,num_group=1, one_layer=False, small_sft=False, blocktype='sft', channel_first=False, drconv=False):
        super().__init__()
        self.drconv=drconv
        self.blocktype=blocktype

        main = []
        # a convolution used to match the channels of the residual blocks
        self.input_conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
 
        if drconv and blocktype=='res':
            self.par_conv = nn.Sequential(*[nn.Conv2d(67, 3, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        elif blocktype=='cbam':
            self.par_conv = nn.Sequential(*[nn.Conv2d(65, out_channels//2, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        else:#sft
            self.par_conv = nn.Sequential(*[nn.Conv2d(1, out_channels//2, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
            
            
        # residual blocks
        if blocktype=='sft':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamicSFT, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight,one_layer=one_layer, small_sft=small_sft)))
        elif blocktype=='res':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamicSFT_res, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight,small_sft=small_sft,channel_first=channel_first,drconv=drconv)))
        elif blocktype=='cbam':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic_cbam, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight,channel_first=channel_first)))
        elif blocktype=='cbam_conv':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic_cbam_conv, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight,channel_first=channel_first)))
        elif blocktype=='dynamic_cbam':
            self.main = nn.Sequential(*(make_layer(ResidualBlockNoBNDynamic_cbam, num_blocks, mid_channels=out_channels,num_experts=num_experts,num_group=num_group, with_bias=with_bias, with_se=with_se, init_weight=init_weight,channel_first=channel_first)))

    def forward(self, feat):# input: lr_curr, key_warp, feat_prop, outputs[i]        
        feat['x'] = self.input_conv(feat['x'])
        if self.blocktype=='cbam':
            feat['par'] = self.par_conv(torch.cat([feat['x'], feat['par']], dim=1))
        elif self.blocktype=='res' or self.blocktype=='sft':
            feat['par'] = self.par_conv(feat['par'])

        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trainmid_channels=out_channels,num_experts=num_experts)ed SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)
