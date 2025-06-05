import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d
# from ops.dcn.deform_conv import ModulatedDeformConvPack
import functools
from torch.autograd import Variable
import numpy as np
import torchvision
from mmedit.models.registry import BACKBONES

from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger

def shape_match(out, out_lst):
        _,_,h,w=out.size()
        _,_,h_l,w_l=out_lst.size()
        if not (h==h_l) or not(w==w_l):
            out = F.interpolate(input=out,size=(h_l, w_l),mode='bilinear',align_corners=False)
        return out 


class FA(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):

        super(FA, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = ModulatedDeformConv2d(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        b, _, _, h, w = inputs.shape
        inputs = inputs.view(b, -1, h, w)


        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out =shape_match(out,out_lst[i])
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )
        fused_feat =shape_match(fused_feat,inputs)
        return fused_feat


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def QE(input, input_nc=64, output_nc=3, ngf=64, n_downsample_global=3, n_blocks_global=9,
             norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    QEnet = QEModule(input, input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        QEnet.cuda(gpu_ids[0])
    QEnet.apply(weights_init)
    return QEnet


class QEModule(nn.Module):
    def __init__(self, input, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(QEModule, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        self.fc = nn.Sequential(
            nn.Linear(4, 512),
            nn.Softplus()
        )
        self.n_blocks = n_blocks
        for i in range(0, n_blocks):
            setattr(self, 'resB' + str(i),
                    ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer))

        self.model = nn.Sequential(*model)

        model2 = [nn.ConvTranspose2d(ngf * 8, int(ngf * 8 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 8 / 2)), activation]
        self.model2 = nn.Sequential(*model2)
        model3 = [nn.ConvTranspose2d(ngf * 4, int(ngf * 4 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 4 / 2)), activation]
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.ConvTranspose2d(ngf * 2, int(ngf * 2 / 2), kernel_size=3, stride=1, padding=1,
                                     output_padding=0),
                  norm_layer(int(ngf * 2 / 2)), activation]
        self.model4 = nn.Sequential(*model4)
        model5 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model5 = nn.Sequential(*model5)

    def forward(self, input, qp_num):
        # breakpoint()
        b, c,_,_,_ = qp_num.size()
        qp_num=qp_num.view(b, c)[:,0] 
        qp = F.one_hot(qp_num.to(torch.int64), 4)
        qp = qp.squeeze(1)
        qp = qp.to(torch.float32)
        qp = self.fc(qp)
        qp = qp.view(b, 512, 1, 1)

        out = self.model(input)
        for i in range(self.n_blocks):
            out = getattr(self, 'resB' + str(i))(out, qp)

        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model2(out)
        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model3(out)
        s1 = 2 * list(out.size())[2]
        s2 = 2 * list(out.size())[3]
        out = nn.functional.interpolate(input=out, size=(s1, s2), mode='bilinear', align_corners=False)
        out = self.model4(out)
        out = self.model5(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

        self.norm_layer = norm_layer(dim)
        self.activation = activation

        self.conv_block2 = self.build_conv_block2(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p)]
        return nn.Sequential(*conv_block)

    def build_conv_block2(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, qp):
        conv_block_out = self.conv_block(x)
        conv_block_out = conv_block_out * qp
        conv_block_out = self.activation(self.norm_layer(conv_block_out))
        conv_block_out = self.conv_block2(conv_block_out)
        out = x + conv_block_out

        return out

@BACKBONES.register_module()
class DCNGAN_Net(nn.Module):
    def __init__(self):
        super(DCNGAN_Net, self).__init__()

        self.in_nc = 3 # for Y channel
        self.radius = 1

        self.FA = FA(
            in_nc=self.in_nc * (2 * self.radius + 1),
            out_nc=64,
            nf=32,
            nb=3,
            deform_ks=3
        )

        self.QE = QE(input)

    def forward(self, x, QPs=None, slices=None, mvs=None,base_QPs=None,par_map=None):
        # breakpoint()
        out = self.FA(x)
        out = self.QE(out, base_QPs)

        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
                            
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


@BACKBONES.register_module()
class discriminator(nn.Module):

    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 0)
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        out = {}
        feature_maps = []
        x1 = F.leaky_relu(self.conv1(input), 0.2)
        feature_maps.append(x1)
        x2 = F.leaky_relu(self.conv2_bn(self.conv2(x1)), 0.2)
        feature_maps.append(x2)
        x3 = F.leaky_relu(self.conv3_bn(self.conv3(x2)), 0.2)
        feature_maps.append(x3)
        x4 = F.leaky_relu(self.conv4_bn(self.conv4(x3)), 0.2)
        feature_maps.append(x4)
        x = self.conv5(x4)
        out['feature_maps'] = feature_maps
        out['prediction'] = x
        return out
    

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

 