# Copyright (c) ryanxingql. All rights reserved.
import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d
from mmcv.runner import load_checkpoint
import torch.nn.functional as F

from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


def shape_match(out, out_lst):
        _,_,h,w=out.size()
        _,_,h_l,w_l=out_lst.size()
        if not (h==h_l) or not(w==w_l):
            out = F.interpolate(input=out,size=(h_l, w_l),mode='bilinear',align_corners=False)
        return out 



class STDF(nn.Module):

    def __init__(self,
                 in_nc=3,
                 out_nc=64,
                 nf=32,
                 nb=3,
                 base_ks=3,
                 deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super().__init__()

        self.in_nc = in_nc
        self.nb = nb
        self.deform_ks = deform_ks

        self.size_dk = deform_ks**2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
        )

        for i in range(1, nb):
            setattr(
                self,
                'dn_conv{}'.format(i),
                nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                ),
            )
            setattr(
                self,
                'up_conv{}'.format(i),
                nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                ),
            )

        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2)

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv2d(
            in_channels=in_nc,
            out_channels=out_nc,
            kernel_size=deform_ks,
            padding=deform_ks // 2,
            deform_groups=in_nc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, _, _, h, w = x.shape
        x = x.view(b, -1, h, w)

        # feature extraction (with down-sampling)
        out_lst = [self.in_conv(x)]  # record feature maps for skip connections
        for i in range(1, self.nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))

        # first down then up at the bottom
        out = self.tr_conv(out_lst[-1])

        # feature reconstruction (with up-sampling)
        for i in range(self.nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out =shape_match(out,out_lst[i])
            out = up_conv(torch.cat([out, out_lst[i]], 1))

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        nc_off = self.in_nc * 2 * self.size_dk
        off = off_msk[:, :nc_off, ...]
        msk = torch.sigmoid(off_msk[:, nc_off:, ...])

        # perform deformable convolutional fusion
        fused_feat = self.relu(self.deform_conv(x, off, msk))
        
        return fused_feat


class QENet(nn.Module):

    def __init__(self, in_nc=64, nf=48, nb=6, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super().__init__()

        padding = base_ks // 2

        self.in_conv = nn.Conv2d(in_nc, nf, base_ks, padding=padding)

        hid_conv_lst = []
        for _ in range(nb):
            hid_conv_lst += [
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, base_ks, padding=padding),
            ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_nc, base_ks, padding=padding),
        )

    def forward(self, x):
        out = self.in_conv(x)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


@BACKBONES.register_module()
class STDFNet(nn.Module):
    """STDF network structure.

    Ref: https://github.com/ryanxingql/stdf-pytorch

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        radius: frames number before the center frame.
        nf_stdf (int): Channel number of intermediate features of STDF module.
            Default: 32.
        nb_stdf (int): Block number of STDF module.
            Default: 3.
        nf_qe (int): Channel number of intermediate features of QE module.
            Default: 48.
        nb_qe (int): Block number of QE module.
            Default: 6.
        deform_ks (int): Kernel size of deformable convolutions.
            Default: 3.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 radius=3,
                 nf_stdf=32,
                 nb_stdf=3,
                 nf_stdf_out=64,
                 deform_ks=3,
                 nf_qe=48,
                 nb_qe=6):

        super().__init__()

        self.radius = radius

        self.nf_qe = nf_qe
        self.nb_qe = nb_qe

        input_len = radius * 2 + 1
        self.stdf = STDF(
            in_nc=in_channels * input_len,
            out_nc=nf_stdf_out,
            nf=nf_stdf,
            nb=nb_stdf,
            deform_ks=deform_ks,
        )

        self.qenet = QENet(
            in_nc=nf_stdf_out,
            nf=nf_qe,
            nb=nb_qe,
            out_nc=out_channels,
        )

    # def forward(self, x):
    def forward(self, x, QPs=None, slices=None, mvs=None,base_QPs=None,par_map=None):

        """Forward function for STDFNet.

        Args:
            x (Tensor): Input tensor with shape (n, t, c, h, w).

        Returns:
            Tensor: Out center frame with shape (n, c, h, w).
        """
        out = self.stdf(x)
        out = self.qenet(out)

        out += x[:, self.radius, ...]  # res: add middle frame
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
