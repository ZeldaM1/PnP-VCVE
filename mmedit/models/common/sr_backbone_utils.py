# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
import torch
import torch.nn.functional as F
from .partition_aware import DRConv2d,SpatialAttention,SpatialAttention_conv,SpatialAttention_simple
 
def default_init_weights_dasr(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)




def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, if_bias=True, K=5, init_weight=False, gaintune=False, gainbias=False):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = if_bias
        self.K = K
        self.gaintune=gaintune
        self.gainbias=gainbias
        if gainbias:
            assert gaintune==True

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes), requires_grad=True)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.if_bias:
                nn.init.constant_(self.bias[i], 0)

    def forward(self, inputs):
        x = inputs['x'] 
        softmax_attention = inputs['weights'] 
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width) 
        weight = self.weight.view(self.K, -1) 
        
        # aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.gaintune:
            aggregate_weight = (aggregate_weight.view(batch_size, in_planes,-1,self.kernel_size, self.kernel_size) * inputs['gamma'].view(batch_size, 1, -1, 1, 1)).view(-1, self.in_planes, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            if self.gainbias:
                aggregate_bias = aggregate_bias + inputs['beta'].contiguous().view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
    
class Dynamic_conv2d_se(Dynamic_conv2d):
    def __init__(self, with_se=False, **kwargs):
        super(Dynamic_conv2d_se, self).__init__(**kwargs)
        self.with_se=with_se
    def forward(self, inputs):
        x = inputs['x'] 
        softmax_attention = inputs['weights'] 
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width) 
        weight = self.weight.view(self.K, -1) 
        
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        if self.with_se:
            output= (output * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) 
        return output
    

class ResidualBlockNoBNDynamic(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_bias=False,with_se=False, init_weight=False,gaintune=False, gainbias=False, one_layer=False, channel_first=False):
        super(ResidualBlockNoBNDynamic, self).__init__()
        self.res_scale = res_scale
        self.with_bias = with_bias
        self.one_layer=one_layer
        self.channel_first=channel_first
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True, groups=num_group) if one_layer else Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,gaintune=gaintune, gainbias=gainbias)
        self.conv2 = Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight)
        self.with_se=with_se

        self.relu = nn.ReLU(inplace=True)
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)
        
 
    def forward(self, inputs):
        identity = inputs['x'].clone()
        if self.channel_first:
            out = self.conv2(inputs)
            if self.with_bias:
                out= (out * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) if self.with_se else (inputs['gamma'].unsqueeze(-1).unsqueeze(-1) * out + inputs['beta'].unsqueeze(-1).unsqueeze(-1))
            out = self.relu(out)
            conv2_input = {'x':out, 'weights':inputs['weights'], 'gamma': inputs['gamma']}
            out = self.conv1(out) if self.one_layer else self.conv1(conv2_input)
        else:
            out = self.relu(self.conv1(inputs['x'])) if self.one_layer else self.relu(self.conv1(inputs))
            conv2_input = {'x':out, 'weights':inputs['weights']}
            out = self.conv2(conv2_input)
            if self.with_bias:
                out= (out * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) if self.with_se else (inputs['gamma'].unsqueeze(-1).unsqueeze(-1) * out + inputs['beta'].unsqueeze(-1).unsqueeze(-1))
        out = identity + out * self.res_scale

        return {'x':out, 'weights':inputs['weights'],'gamma': inputs['gamma'], 'beta': inputs['beta']}


def mask_roi(feature, size, kernel_size=1):
    (h_idx, w_idx)=size
    if kernel_size == 1:
        return feature[0, :, h_idx, w_idx]
    else:
        assert TypeError("support kernel=1 only")
 
def mask_roi_back(feature, roi_feature, size, kernel_size=1):
    (h_idx, w_idx)=size
    if kernel_size == 1:
        feature[0, :, h_idx, w_idx]=roi_feature
    else:
        assert TypeError("support kernel=1 only")
    return feature


class ResidualBlockNoBNDynamic_drt(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_se=False, init_weight=False, one_layer=False,channel_first=True, sparse_val=False):
        super(ResidualBlockNoBNDynamic_drt, self).__init__()
        self.res_scale = res_scale
        self.one_layer=one_layer
        self.channel_first=channel_first
        self.sparse_val=sparse_val
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True, groups=num_group) if one_layer else Dynamic_conv2d_se(in_planes=mid_channels, out_planes=mid_channels, kernel_size=3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,with_se=with_se)
        self.conv2 = Dynamic_conv2d_se(in_planes=mid_channels, out_planes=mid_channels, kernel_size=3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,with_se=with_se)
        self.conv16x16 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.conv16x8 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.conv8x8 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.relu = nn.ReLU(inplace=True)
        for m in [self.conv1, self.conv2, self.conv16x16, self.conv16x8, self.conv8x8]:
            default_init_weights(m, 0.1)

    def sparse_conv(self,feature,inputs):
        res_16x16 = torch.mm(self.conv16x16.weight.view(64, -1), mask_roi(feature, inputs['sparse_0'])) 
        res_16x8 = torch.mm(self.conv16x8.weight.view(64, -1), mask_roi(feature, inputs['sparse_1'])) 
        res_8x8 = torch.mm(self.conv8x8.weight.view(64, -1), mask_roi(feature, inputs['sparse_2'])) 
        dyres= torch.zeros_like(feature)
        dyres= mask_roi_back(dyres,res_16x16, inputs['sparse_0'])
        dyres= mask_roi_back(dyres,res_16x8, inputs['sparse_1']) 
        dyres= mask_roi_back(dyres,res_8x8, inputs['sparse_2'])
        return dyres/255
 
    def forward(self, inputs):
        identity = inputs['x'].clone()
        if self.channel_first:
            if self.sparse_val and (not self.training):
                dyres= self.sparse_conv(inputs['x'], inputs)
            else:
                dyres= self.conv16x16(inputs['x'])*inputs['par'][:,0,...]  + self.conv16x8(inputs['x'])*inputs['par'][:,1,...]  + self.conv8x8(inputs['x'])*inputs['par'][:,2,...]
            out = self.relu(self.conv2(inputs) + dyres)
            conv2_input = {'x':out, 'weights':inputs['weights'], 'gamma': inputs['gamma']}
            out = self.conv1(out) if self.one_layer else self.conv1(conv2_input)
        else:
            out = self.relu(self.conv1(inputs['x'])) if self.one_layer else self.relu(self.conv1(inputs))
            conv2_input = {'x':out, 'weights':inputs['weights'], 'gamma': inputs['gamma']}
            # import time
            # torch.cuda.synchronize()
            # start=time.time()
            if self.sparse_val and (not self.training):  
                dyres= self.sparse_conv(out, inputs)
            else:
                dyres= self.conv16x16(out)*inputs['par'][:,0,...]  + self.conv16x8(out)*inputs['par'][:,1,...]  + self.conv8x8(out)*inputs['par'][:,2,...]
            # torch.cuda.synchronize()
            # end=time.time()
            # print('---------------------',end-start)
            out = self.conv2(conv2_input) + dyres
        
        out = identity + out * self.res_scale

        # return {'x':out, 'par': inputs['par'], 'weights':inputs['weights'],'gamma': inputs['gamma'], 'beta': inputs['beta']}
        inputs['x']=out
        return inputs


class ResidualBlockNoBNDynamic_drt_wo_qp(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_se=False, init_weight=False, one_layer=False,channel_first=True, sparse_val=False):
        super(ResidualBlockNoBNDynamic_drt_wo_qp, self).__init__()
        self.res_scale = res_scale
        self.one_layer=one_layer
        self.channel_first=channel_first
        self.sparse_val=sparse_val
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True, groups=num_group) if one_layer else Dynamic_conv2d_se(in_planes=mid_channels, out_planes=mid_channels, kernel_size=3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,with_se=with_se)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True, groups=num_group) if one_layer else Dynamic_conv2d_se(in_planes=mid_channels, out_planes=mid_channels, kernel_size=3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,with_se=with_se)

        self.conv16x16 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.conv16x8 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.conv8x8 = nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False, groups=num_group)
        self.relu = nn.ReLU(inplace=True)
        for m in [self.conv1, self.conv2, self.conv16x16, self.conv16x8, self.conv8x8]:
            default_init_weights(m, 0.1)

    def sparse_conv(self,feature,inputs):
        res_16x16 = torch.mm(self.conv16x16.weight.view(64, -1), mask_roi(feature, inputs['sparse_0'])) 
        res_16x8 = torch.mm(self.conv16x8.weight.view(64, -1), mask_roi(feature, inputs['sparse_1'])) 
        res_8x8 = torch.mm(self.conv8x8.weight.view(64, -1), mask_roi(feature, inputs['sparse_2'])) 
        dyres= torch.zeros_like(feature)
        dyres= mask_roi_back(dyres,res_16x16, inputs['sparse_0'])
        dyres= mask_roi_back(dyres,res_16x8, inputs['sparse_1']) 
        dyres= mask_roi_back(dyres,res_8x8, inputs['sparse_2'])
        return dyres/255
 
    def forward(self, inputs):
        identity = inputs['x'].clone()
        if self.channel_first:
            if self.sparse_val and (not self.training):
                dyres= self.sparse_conv(inputs['x'], inputs)
            else:
                dyres= self.conv16x16(inputs['x'])*inputs['par'][:,0,...]  + self.conv16x8(inputs['x'])*inputs['par'][:,1,...]  + self.conv8x8(inputs['x'])*inputs['par'][:,2,...]
            out = self.relu(self.conv2(inputs['x']) + dyres)
            out = self.conv1(out)  
        else:
            out = self.relu(self.conv1(inputs['x']))  
            if self.sparse_val and (not self.training):  
                dyres= self.sparse_conv(out, inputs)
            else:
                dyres= self.conv16x16(out)*inputs['par'][:,0,...]  + self.conv16x8(out)*inputs['par'][:,1,...]  + self.conv8x8(out)*inputs['par'][:,2,...]
            out = self.conv2(out) + dyres
        
        out = identity + out * self.res_scale

        inputs['x']=out
        return inputs



class SFTLayer(nn.Module):
    def __init__(self, mid_channels=64,small_sft=False,init_weight=False):
        super(SFTLayer, self).__init__()
        self.small_sft=small_sft
        self.SFT_scale_conv0 = nn.Conv2d(mid_channels//2+mid_channels, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(mid_channels//2+mid_channels, 64, 1)
        if not small_sft:
            self.SFT_scale_conv1 = nn.Conv2d(mid_channels, mid_channels, 1)
            self.SFT_shift_conv1 = nn.Conv2d(mid_channels, mid_channels, 1)
        # breakpoint()
        if init_weight:
            init_list=[self.SFT_scale_conv0, self.SFT_shift_conv0]
            if not small_sft:
                init_list.append(self.SFT_scale_conv1, self.SFT_shift_conv1)

            for m in init_list:
                default_init_weights(m, 0.1)

    def forward(self, feas, side_feas):
        x_in = torch.cat([feas, side_feas],1)
        if not self.small_sft:
            scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x_in), 0.1, inplace=True))
            shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x_in), 0.1, inplace=True))
        else:
            scale = self.SFT_scale_conv0(x_in) 
            shift = self.SFT_shift_conv0(x_in) 
        # breakpoint()
        # return feas * (scale + 1) + shift
        return feas * scale + shift
 

class ResidualBlockNoBNDynamicSFT(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_bias=False,with_se=False, init_weight=False,gaintune=False, gainbias=False, one_layer=False, small_sft=False):
        super(ResidualBlockNoBNDynamicSFT, self).__init__()
        self.res_scale = res_scale
        self.with_bias = with_bias
        self.one_layer=one_layer
        self.sft1 = SFTLayer(mid_channels,small_sft)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True, groups=num_group) if one_layer else Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight,gaintune=gaintune, gainbias=gainbias)
        self.sft2 = SFTLayer(mid_channels,small_sft)
        self.conv2 = Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight)
        self.with_se=with_se

        self.relu = nn.ReLU(inplace=True)
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)
        
 
    def forward(self, inputs):
        identity = inputs['x'].clone()
        sft_out1 = self.sft1(inputs['x'], inputs['par'])
        conv2_input = {'x':sft_out1, 'weights':inputs['weights']}
        out = self.relu(self.conv1(sft_out1)) if self.one_layer else self.relu(self.conv1(conv2_input))
        sft_out2 = self.sft2(out, inputs['par'])
        conv2_input = {'x':sft_out2, 'weights':inputs['weights']}
        out = self.conv2(conv2_input)
        if self.with_bias:
            out= (out * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) if self.with_se else (inputs['gamma'].unsqueeze(-1).unsqueeze(-1) * out + inputs['beta'].unsqueeze(-1).unsqueeze(-1))
        out = identity + out * self.res_scale

        return {'x':out, 'par': inputs['par'], 'weights':inputs['weights'],'gamma': inputs['gamma'], 'beta': inputs['beta']}
    

class ResidualBlockNoBNDynamicSFT_res(nn.Module):
    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_bias=False,with_se=False, init_weight=False,small_sft=False,channel_first=False,drconv=False):
        super(ResidualBlockNoBNDynamicSFT_res, self).__init__()
        self.res_scale = res_scale
        self.with_bias = with_bias
        self.channel_first=channel_first
        self.sft1 = DRConv2d(mid_channels, mid_channels, kernel_size=3, padding=1, region_num=3) if drconv else SFTLayer(mid_channels,small_sft,init_weight=init_weight)
        self.conv1 = Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight)
        self.with_se=with_se

        self.relu = nn.ReLU(inplace=True)
        # for m in [self.sft1, self.conv1]:
        #     default_init_weights(m, 0.1)
        
 
    def forward(self, inputs):
        identity = inputs['x'].clone()
        if self.channel_first:
            out = self.relu(self.conv1(inputs))
            if self.with_bias:
                out= (out * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) if self.with_se else (inputs['gamma'].unsqueeze(-1).unsqueeze(-1) * out + inputs['beta'].unsqueeze(-1).unsqueeze(-1))
            out = self.sft1(out, inputs['par'])
        else:# spatial (sft1)  -> channel (conv1)
            sft_out1 = self.sft1(inputs['x'], inputs['par'])
            conv2_input = {'x':sft_out1, 'weights':inputs['weights']}
            out = self.relu(self.conv1(conv2_input))
            if self.with_bias:
                out= (out * inputs['gamma'].unsqueeze(-1).unsqueeze(-1)) if self.with_se else (inputs['gamma'].unsqueeze(-1).unsqueeze(-1) * out + inputs['beta'].unsqueeze(-1).unsqueeze(-1))

        out = identity + out * self.res_scale

        return {'x':out, 'par': inputs['par'], 'weights':inputs['weights'],'gamma': inputs['gamma'], 'beta': inputs['beta']}
    

class ResidualBlockNoBNDynamic_cbam(ResidualBlockNoBNDynamicSFT_res):
    def __init__(self, mid_channels=64, res_scale=1, num_experts=10, num_group=1, with_bias=False,with_se=False, init_weight=False,channel_first=False):
        super(ResidualBlockNoBNDynamic_cbam, self).__init__()
        self.res_scale = res_scale
        self.with_bias = with_bias
        self.channel_first=channel_first
        self.with_se=with_se
        self.sft1 = SpatialAttention(mid_channels//2, kernel_size=3, padding=1, init_weight=init_weight)
        self.conv1 = Dynamic_conv2d(mid_channels, mid_channels, 3, groups=num_group, if_bias=True, K=num_experts, init_weight=init_weight)
        self.relu = nn.ReLU(inplace=True)

class ResidualBlockNoBNDynamic_cbam_conv(ResidualBlockNoBNDynamic_cbam):
    def __init__(self, mid_channels=64, init_weight=False, num_group=1,num_experts=10, **kwargs):
        super(ResidualBlockNoBNDynamic_cbam_conv, self).__init__(mid_channels=mid_channels,init_weight=init_weight,num_group=num_group, num_experts=num_experts,**kwargs)
        self.sft1 = SpatialAttention_conv(mid_channels, kernel_size=3, padding=1, init_weight=init_weight)

