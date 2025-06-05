# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
import math
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from .FBCNN_QFA import QFAttention


class Jpeg_domain(nn.Module):
    def __init__(self, n_atten=4, use_base_qp=False, **kwargs):
        super().__init__(**kwargs)
        self.n_atten=n_atten
        self.use_base_qp=use_base_qp
        self.qf_embed = nn.Sequential(*[nn.Linear(1, 64),
                                #   nn.ReLU(),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Linear(64, 64),
                                #   nn.ReLU(),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                  nn.Linear(64, 64),
                                #   nn.ReLU()
                                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                ]
                                )
        self.to_gamma = nn.Sequential(*[nn.Linear(64, 64),nn.Sigmoid()])
        self.to_beta =  nn.Sequential(*[nn.Linear(64, 64),nn.Tanh()])
        self.m_up = nn.Sequential(*[QFAttention(64,64, bias=True) for _ in range(n_atten)])
     
    def forward(self, feat_prop, used_QPs, base_QPs):#used_QPs[:, i, :, :, :].view(n,-1)
        qf_embedding = self.qf_embed(base_QPs) if self.use_base_qp else self.qf_embed(used_QPs)
        gamma = self.to_gamma(qf_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(qf_embedding).unsqueeze(-1).unsqueeze(-1) 
        for atten_idx in range(self.n_atten):
            feat_prop = self.m_up[atten_idx](feat_prop, gamma, beta)
        return feat_prop



class one_for_all_domain(nn.Module):
    def __init__(self, use_base_qp=False, **kwargs):
        super().__init__(**kwargs)
        self.use_base_qp=use_base_qp
        self.weight_U = nn.Sequential(*[nn.Linear(10, 64), nn.LeakyReLU(negative_slope=0.1, inplace=True)])
        
    def forward(self, feat_prop, QP, base_QPs):
        b,c,_,_=feat_prop.shape
        vector = F.one_hot((QP/10).floor().to(torch.int64),num_classes=10).to(torch.float32)
        attention_map=torch.sigmoid(self.weight_U(vector)).view(b,-1,1,1)
        feat_prop= attention_map * feat_prop

        return feat_prop



class QENET(nn.Module):
    def __init__(self, in_nc=64, nf=64, out_nc=1, base_ks=3, use_base_qp=False):
        super(QENET, self).__init__()
        self.use_base_qp=use_base_qp
        self.one_hot = F.one_hot
        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.Softplus()
        )

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.hid_conv1 = nn.Conv2d(in_nc, nf, base_ks, padding=1)
        self.hid_conv2 = nn.Conv2d(nf, nf, base_ks, padding=1)

    def forward(self, feat_prop, used_QPs, base_QPs):#inputs, qp
        
        QP = base_QPs if self.use_base_qp else used_QPs
        b, c = QP.size()
        x = F.one_hot((QP/10).floor().to(torch.int64),num_classes=10).to(torch.float32)
        x = x.squeeze(1)
        x = x.to(torch.float32)
        x = self.fc(x)
        x = x.view(b, 64, 1, 1)

        out1 = self.relu(self.hid_conv1(feat_prop) * x)
        out2 = self.relu(self.hid_conv2(out1) * x)
        # breakpoint()
        return out2




class ScaleAwareConv(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, num_experts= 4):
        super(ScaleAwareConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.num_experts = num_experts
        assert num_experts >1 

        # Use fc layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Linear(64, num_experts),
            nn.Softmax(1)
        )
 
        # Initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, out_channels))
            # Calculate fan_in
            dimensions = self.weight_pool.dim()
            if dimensions < 2:
                raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

            num_input_feature_maps = self.weight_pool.size(1)
            receptive_field_size = 1
            if self.weight_pool.dim() > 2:
                # math.prod is not always available, accumulate the product manually
                # we could use functools.reduce but that is not supported by TorchScript
                for s in self.weight_pool.shape[2:]:
                    receptive_field_size *= s
            fan_in = num_input_feature_maps * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, feat_props, QPs): 
        b,c,_,_=feat_props.shape
        outs=[]

        for feat_prop,QP in zip(feat_props, QPs):
            feat_prop=feat_prop.unsqueeze(0)
            QP=QP.unsqueeze(0)
            
            routing_weights = self.routing(QP).view(self.num_experts, 1, 1)
            fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
            fused_weight = fused_weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)
            if self.bias:
                fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
            else:
                fused_bias = None
            out = F.conv2d(feat_prop, fused_weight, fused_bias, self.stride, self.padding)
            outs.append(out)
        return torch.cat(outs, dim=0)



        #  def forward(self, inputs):
        # b, _, _, _ = inputs.size()
        # res = []
        # for input in inputs:
        #     input = input.unsqueeze(0)
        #     pooled_inputs = self._avg_pooling(input)
        #     routing_weights = self._routing_fn(pooled_inputs)
        #     kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
        #     out = self._conv_forward(input, kernels)
        #     res.append(out)
        # return torch.cat(res, dim=0)

class Base_Predictor(nn.Module):
    def __init__(self, nf=64, num_experts=5, softmax=False):
        super(Base_Predictor, self).__init__()   
        route=[nn.Linear(1, nf), nn.ReLU(True), nn.Linear(nf, num_experts)]
        if softmax:
            route.append(nn.Softmax(1))
        self.BaseNet = nn.Sequential(*route)
 
    def forward(self, CRFs):
        b,t,_,_,_=CRFs.shape 
        mapped_weights = self.BaseNet(CRFs.view(-1,1)).view(b,t,-1) 
        return mapped_weights

class Bias_Predictor(nn.Module):
    def __init__(self, nf=64, with_bias=True):
        super(Bias_Predictor, self).__init__()
        self.with_bias=with_bias
        self.qf_embed = nn.Sequential(*[nn.Linear(1, nf),nn.ReLU(True)])
        self.to_gamma = nn.Sequential(*[nn.Linear(nf, nf),nn.Sigmoid()])
        if with_bias:
            self.to_beta =  nn.Sequential(*[nn.Linear(nf, nf),nn.Tanh()])
 
    def forward(self, QPs):
        b,t,_,_,_=QPs.shape
        qf_embedding = self.qf_embed(QPs.view(-1,1)).view(b,t,-1)
        gamma = self.to_gamma(qf_embedding) 
        beta = self.to_beta(qf_embedding) if self.with_bias else None
        return gamma, beta

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 3.


class SEModule(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(SEModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )
    def forward(self, QPs):
        b,t,_,_,_=QPs.shape
        gamma = self.fc(QPs.view(-1,1)).view(b,t,-1)
        return gamma, None
 
