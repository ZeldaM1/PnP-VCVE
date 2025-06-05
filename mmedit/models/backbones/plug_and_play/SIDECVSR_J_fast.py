import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


def nd_meshgrid(h, w, device):
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
    return torch.from_numpy(id_flow).float().to(device)


class STN(nn.Module):
    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize
    def forward(self, inputs, u, v):
        mesh = nd_meshgrid(h = inputs.shape[2], w = inputs.shape[3], device = inputs.device)
        if not self.norm:
            h, w = inputs.shape[-2:]
            _u = (u / w * 2) * 32
            _v = (v / h * 2) * 32
        flow = torch.stack([_u, _v], dim=-1).to('cuda')
        mesh = (mesh + flow).clamp(-1,1)
        # warped_img = F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode) ### original 1.1.0
        warped_img = F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)
        return warped_img


class MV_LOCAL_ATTN(nn.Module):

    def __init__(self, nf=64, p_k=3):
        super(MV_LOCAL_ATTN, self).__init__()
        self.nf = nf
        self.make_fea_patches = torch.nn.Unfold(kernel_size=(p_k, p_k), padding=p_k//2, stride=1)
        self.warp_module = STN(padding_mode='border', normalize=False)

        self.kernel_pred_module = nn.Sequential(
            nn.Conv2d(nf * p_k * p_k * 2, nf, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, p_k * p_k, 1, 1, 0, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, nbh_fea, cen_fea, mv):
        B, C, H, W = cen_fea.shape
        nbh_fea_p = self.make_fea_patches(nbh_fea)
        nbh_fea_p = nbh_fea_p.view(B, -1, H, W)
        
        cen_fea_p = self.make_fea_patches(cen_fea)
        cen_fea_p = cen_fea_p.view(B, -1, H, W)

        aligned_nbh_fea_p = self.warp_module(nbh_fea_p, mv[:,0,:,:], mv[:,1,:,:])  # aligned_nbh_fea_p.shape = (B, 64*9, H, W)
        fuse_fea = torch.cat([aligned_nbh_fea_p, cen_fea_p], 1)
        local_attn_map = self.kernel_pred_module(fuse_fea)   # (B, 9, H, W)
        
        aligned_nbh_fea_p = aligned_nbh_fea_p.view(B, C, -1, H, W) 
        local_attn_map = torch.unsqueeze(local_attn_map, 1)
        alg_attn_nbh_fea = torch.mean(aligned_nbh_fea_p * local_attn_map, 2)

        return alg_attn_nbh_fea.view(B, -1, H, W)


class SIDECVSR(nn.Module):

    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(SIDECVSR, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining

        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.feature_extraction = side_embeded_feature_extract_block()

        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = SCNet(nf=nf, SCGroupN=SCGs)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf + nf // 4 + nf // 16, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.mv_patch_attn = MV_LOCAL_ATTN(nf=nf)

        #### fea fusion attn
        self.tmp_fea_attn = fea_fusion(nf=nf)

        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)

        #### fea pyramid fuse 
        self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

        #### 
        self.side_fea_ext = side_to_fea()


    def forward(self, x, mvs, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        x_center = x[:, self.center, :, :, :].contiguous()
        # sides = torch.cat([rms.view(-1, C, H, W), pms.view(-1, C, H, W), ufs.view(-1, C, H, W)], 1)
        # sides_fea = self.side_fea_ext(sides)
        
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides = torch.cat([rms.view(-1, C, H, W), pms.view(-1, C, H, W), ufs.view(-1, C, H, W)], 1)
            sides_fea = self.side_fea_ext(sides)
            L1_fea = self.feature_extraction(L1_fea, sides_fea)
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides = torch.cat([rms[:,-1,:,:,:], pms[:,-1,:,:,:], ufs[:,-1,:,:,:]], 1)
            need_add_sides_fea = self.side_fea_ext(need_add_sides)

            need_add_L1_fea = self.feature_extraction(need_add_fea, need_add_sides_fea)
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)

            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        
        feas_pyr.append(L1_fea)
        L2_fea = self.down(L1_fea)
        feas_pyr.append(L2_fea)
        L3_fea = self.down(L2_fea)
        feas_pyr.append(L3_fea)

        fuse_fea_pyr = []
        for pyr_i in range(3):
            fea_one_lv = feas_pyr[pyr_i].view(B, N, -1, H//(2**pyr_i), W//(2**pyr_i))
            # local attention
            aligned_fea = []
            for i in range(N):
                if i != N // 2:
                    if pyr_i == 0:
                        tmp_mv = mvs[:,i,:,:,:].clone()
                    if pyr_i == 1:
                        tmp_mv = F.interpolate(mvs[:,i,:,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0
                    if pyr_i == 2:
                        tmp_mv = F.interpolate(mvs[:,i,:,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0
                     
                    alg_nbh_fea = self.mv_patch_attn(fea_one_lv[:,i,:,:,:].clone(), fea_one_lv[:, N//2,:,:,:].clone(), tmp_mv) ### original mv
                    aligned_fea.append(alg_nbh_fea)
                else:
                    aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())

            # feature fusion
            aligned_fea = torch.stack(aligned_fea, dim=1)                      # [B, N, C, H, W]
            aligned_fea = aligned_fea.view(B, -1, H//(2**pyr_i), W//(2**pyr_i))
            
            fea = self.lrelu(self.tsa_fusion(self.tmp_fea_attn(aligned_fea)))  ### tmp_attn + fusion

            fuse_fea_pyr.append(fea)

        # reconstruct
        out = self.recon_trunk(fuse_fea_pyr)

        out_L3 = self.lrelu(self.upconv1_L3(out[2]))
        out_L3 = self.pixel_shuffle(self.pixel_shuffle(out_L3))
        out_L2 = self.lrelu(self.upconv1_L2(out[1]))
        out_L2 = self.pixel_shuffle(out_L2)
        out_fuse = torch.cat([out[0], out_L2, out_L3], 1)
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out_fuse)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class fea_fusion(nn.Module):
    def __init__(self, nf=64):
        super(fea_fusion, self).__init__()

        self.tAtt = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.N = 7
        self.nf = nf
    
    def forward(self, feas):
        B, _, H, W = feas.size()
        emb = self.tAtt(feas.view(-1, self.nf, H, W)).view(B, self.N, -1, H, W)
        emb_ref = emb[:, self.N//2, :, :, :].contiguous()
        cor_l = []
        for i in range(self.N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)

        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, self.nf, 1, 1).view(B, -1, H, W)
        feas_ = feas * cor_prob

        return feas_


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class Block(nn.Module):

    def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               group=4):
        super(Block, self).__init__()

        body = []
        conv = nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2)
        body.append(conv)
        body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv = nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2)
        body.append(conv)
        initialize_weights(body, 0.1)
        self.body = nn.Sequential(*body)

        down = []
        down.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        down.append(Interpolate(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        up.append(Interpolate(scale_factor=2.0))
        self.up = nn.Sequential(*up)
        initialize_weights([self.up, self.down], 0.1)

    def forward(self, x_list):
        res_list = [self.body(x) for x in x_list]
        down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
        up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
        x_list = [
            x + r + d + u
            for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
        ]
        return x_list


class SCGroup(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(SCGroup, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(
                Block(
                    nf,
                    kernel_size=3,
                    width_multiplier=2
                ))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        res_list = [self.conv(x) for x in res_list]
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list


class SCNet(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(SCNet, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(SCGroup())
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32+64, 64, 1)
        self.SFT_scale_conv1 = nn.Conv2d(64, 64, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32+64, 64, 1)
        self.SFT_shift_conv1 = nn.Conv2d(64, 64, 1)

    def forward(self, feas, side_feas):
        x_in = torch.cat([feas, side_feas],1)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x_in), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x_in), 0.1, inplace=True))
        return feas * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, feas, side_feas):
        fea = self.sft0(feas, side_feas)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1(fea, side_feas)
        fea = self.conv1(fea)
        return feas + fea  # return a tuple containing features and conditions


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
