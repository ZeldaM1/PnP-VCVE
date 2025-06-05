# Written by Jinghao Zhou
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable, Function
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.

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

class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)
    
    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2) # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1) # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True)) # B x 3 x 25 x 25
        return grad_kernel, grad_guide


def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk, **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)


class SpatialAttention(nn.Module):
    def __init__(self, input_ch, kernel_size=3, padding=1, init_weight=False):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        if init_weight:
            default_init_weights(self.conv1)

    def forward(self, x, par_x):
        # breakpoint()
        avg_out = torch.mean(par_x, dim=1, keepdim=True)
        max_out, _ = torch.max(par_x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)      
        return (self.sigmoid(out) * x)

class SpatialAttention_conv(nn.Module):
    def __init__(self, input_ch, kernel_size=3, padding=1, init_weight=False):
        super(SpatialAttention_conv, self).__init__()
        self.conv1 = nn.Conv2d(input_ch+2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Hsigmoid()
        if init_weight:
            default_init_weights(self.conv1)

    def forward(self, x, par_x):
        avg_out = torch.mean(par_x, dim=1, keepdim=True)
        max_out, _ = torch.max(par_x, dim=1, keepdim=True)
        atten = self.conv1(torch.cat([x,avg_out,max_out], dim=1))
        atten = self.sigmoid(atten)
        return (atten * x)

 
class SpatialAttention_simple(nn.Module):
    def __init__(self, input_ch, kernel_size=1, padding=0, init_weight=False):
        super(SpatialAttention_simple, self).__init__()
        self.conv1 = nn.Conv2d(input_ch+1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Hsigmoid()
        if init_weight:
            default_init_weights(self.conv1)

    def forward(self, x, par_x):
        atten = self.conv1(torch.cat([x,par_x], dim=1)) 
        atten = self.sigmoid(atten)
        return (atten * x)
    

class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=3, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = region_num

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        self.conv_guide = nn.Conv2d(4, region_num, kernel_size=kernel_size, **kwargs)
   

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.asign_index = asign_index.apply

        for m in self.conv_kernel:
            default_init_weights(m, 0.1)
        default_init_weights(self.conv_guide)

    def forward(self, input, par_map):#input: lr_curr, key_warp, feat_prop, outputs[i]
        # breakpoint()
        kernel = self.conv_kernel(input)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H
        guide_feature = self.conv_guide(torch.cat([input,par_map],dim=1))
        output = self.asign_index(output, guide_feature)
        return output
 

 
class side_to_fea(nn.Module):
    def __init__(self, nf=32):
        super(side_to_fea, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)


class side_embeded_feature_extract_block(nn.Module):
    def __init__(self, nf=64):
        super(side_embeded_feature_extract_block, self).__init__()

        self.RB_wSide_1 = ResBlock_SFT()
        self.RB_wSide_2 = ResBlock_SFT()
        self.RB_wSide_3 = ResBlock_SFT()
        self.RB_wSide_4 = ResBlock_SFT()
        self.RB_wSide_5 = ResBlock_SFT()
        self.RB_wSide_6 = ResBlock_SFT()
        self.RB_wSide_7 = ResBlock_SFT()


    def forward(self, img_feas, side_feas):
        fea1_o = self.RB_wSide_1(img_feas, side_feas)
        fea2_o = self.RB_wSide_2(fea1_o, side_feas)
        fea3_o = self.RB_wSide_3(fea2_o, side_feas)
        fea4_o = self.RB_wSide_4(fea3_o, side_feas)
        fea5_o = self.RB_wSide_5(fea4_o, side_feas)
        fea6_o = self.RB_wSide_6(fea5_o, side_feas)
        fea7_o = self.RB_wSide_7(fea6_o, side_feas)
        
        return fea7_o

class side_embeded_feature_extract_block(nn.Module):
    def __init__(self, nf=64):
        super(side_embeded_feature_extract_block, self).__init__()

        self.RB_wSide_1 = ResBlock_SFT()
        self.RB_wSide_2 = ResBlock_SFT()
        self.RB_wSide_3 = ResBlock_SFT()
        self.RB_wSide_4 = ResBlock_SFT()
        self.RB_wSide_5 = ResBlock_SFT()
        self.RB_wSide_6 = ResBlock_SFT()
        self.RB_wSide_7 = ResBlock_SFT()


    def forward(self, img_feas, side_feas):
        fea1_o = self.RB_wSide_1(img_feas, side_feas)
        fea2_o = self.RB_wSide_2(fea1_o, side_feas)
        fea3_o = self.RB_wSide_3(fea2_o, side_feas)
        fea4_o = self.RB_wSide_4(fea3_o, side_feas)
        fea5_o = self.RB_wSide_5(fea4_o, side_feas)
        fea6_o = self.RB_wSide_6(fea5_o, side_feas)
        fea7_o = self.RB_wSide_7(fea6_o, side_feas)
        
        return fea7_o


#     self.side_fea_ext = side_to_fea()
#     self.feature_extraction = side_embeded_feature_extract_block()

# need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
# need_add_sides = torch.cat([rms[:,-1,:,:,:], pms[:,-1,:,:,:], ufs[:,-1,:,:,:]], 1)
# need_add_sides_fea = self.side_fea_ext(need_add_sides)

# need_add_L1_fea = self.feature_extraction(need_add_fea, need_add_sides_fea)
# need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
# pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
# L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)




def generate_indices(spa_mask, kernel_size):
    A = torch.arange(3).to(spa_mask.device).view(-1, 1, 1)
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
 
 
class Sparse_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, if_bias=False, K=5, init_weight=False, gaintune=False, gainbias=False):
        super(Sparse_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.kernel_size = kernel_size
        self.sparse_conv = nn.Conv2d(in_planes, out_channels, kernel_size, stride, padding, bias=bias)
        if init_weight:
            default_init_weights(self.sparse_conv, 0.1)
        self.kernel_d2s = self.sparse_conv.weight.view(out_channels, -1)

    def mask_select(self, feature, h_idx, w_idx):
        if self.kernel_size == 1:
            return feature[0, :, h_idx, w_idx]
        if self.kernel_size == 3:
            return F.pad(feature, [1, 1, 1, 1])[0, :, h_idx, w_idx].view(9 * feature.size(1), -1)
    
    def forward(feature, h_idx, w_idx):
        sparse_feature = mask_select()
        fea_d2s = torch.mm(self.kernel_d2s, self._mask_select(fea_dense, k))
        return fea_d2s
 
 