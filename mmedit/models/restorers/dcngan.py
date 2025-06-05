# Copyright (c) ryanxingql. All rights reserved.
import numbers
import os.path as osp
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import mmcv
from ..common import set_requires_grad

from mmedit.core import psnr, ssim, tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer
from ..builder import build_backbone, build_loss,build_component

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class ImagePyramide(torch.nn.Module):

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

@MODELS.register_module()
class DCNGAN(BasicRestorer):
    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(BasicRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        # discriminator
        self.discriminator = build_component(
            discriminator) if discriminator else None

        # support fp16
        self.fp16_enabled = False

        # loss
        self.gan_loss = build_loss(gan_loss) if gan_loss else None
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

        self.init_weights(pretrained) 

        if self.perceptual_loss:
            self.scales = [1, 0.5, 0.25, 0.125]
            self.pyramid = ImagePyramide(self.scales, num_channels=3)
        

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1
        

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained=pretrained)
        if self.discriminator:
            self.discriminator.init_weights(pretrained=pretrained)

            

 
    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """

        # during initialization, load weights from the ema model
        if (self.step_counter == self.start_iter
                and self.generator_ema is not None):
            if is_module_wrapper(self.generator):
                self.generator.module.load_state_dict(
                    self.generator_ema.module.state_dict())
            else:
                self.generator.load_state_dict(self.generator_ema.state_dict())

        # data
        lq = data_batch['lq']
        gt = data_batch['gt']
        base_QPs= data_batch['base_QPs']

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()

 
        # generator
        fake_g_output = self.generator(lq,base_QPs=base_QPs)
        losses = dict()
        log_vars = dict()

        # reshape: (n, t, c, h, w) -> (n*t, c, h, w)
        
        c, h, w = gt.shape[1:]
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)
        # breakpoint()
        fake_g_output = fake_g_output.view(-1, c, h, w)

        # no updates to discriminator parameters
        if self.gan_loss:
            set_requires_grad(self.discriminator, False)

        if (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps):
            if self.pixel_loss:
                losses['loss_pix'] = self.pixel_loss(fake_g_output, gt_pixel)
             
            if self.perceptual_loss:
                pyramide_real = self.pyramid(gt_percep)
                pyramide_generated = self.pyramid(fake_g_output)
                vgg_loss = 0
                for scale in self.scales:
                    loss_percep, loss_style = self.perceptual_loss(pyramide_generated['prediction_' + str(scale)], pyramide_real['prediction_' + str(scale)])
                    vgg_loss +=  loss_percep

 


                if loss_percep is not None:
                    losses['loss_perceptual'] = vgg_loss
                if loss_style is not None:
                    losses['loss_style'] = loss_style
            # breakpoint()
            # gan loss for generator
            if self.gan_loss:
                fake_g_pred = self.discriminator(fake_g_output)['prediction']
                losses['loss_gan'] = self.gan_loss(
                    fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self.parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        if self.gan_loss:
            set_requires_grad(self.discriminator, True)
            # real
            real_d_pred = self.discriminator(gt_gan)['prediction']
            loss_d_real = self.gan_loss(
                real_d_pred, target_is_real=True, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_real=loss_d_real))
            optimizer['discriminator'].zero_grad()
            loss_d.backward()
            log_vars.update(log_vars_d)

            # fake
            fake_d_pred = self.discriminator(fake_g_output.detach())['prediction']
            loss_d_fake = self.gan_loss(
                fake_d_pred, target_is_real=False, is_disc=True)
            loss_d, log_vars_d = self.parse_losses(
                dict(loss_d_fake=loss_d_fake))
            loss_d.backward()
            log_vars.update(log_vars_d)

            optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=fake_g_output.cpu()))

        return outputs

 
    def forward_test(self, lq, gt=None, QPs=None, slices=None, mvs=None, base_QPs=None, par_map=None, meta=None, save_image=False, save_path=None, iteration=None):

        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq,base_QPs=base_QPs)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            gt_path = meta[0]['gt_path'][0]
            folder_name = meta[0]['key'].split('/')[0]
            frame_name = osp.splitext(osp.basename(gt_path))[0]

            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{frame_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, folder_name,
                                     f'{frame_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results
