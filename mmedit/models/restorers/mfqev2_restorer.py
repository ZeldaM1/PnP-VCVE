# Copyright (c) ryanxingql. All rights reserved.
import numbers
import os.path as osp

import mmcv
import torch

from mmedit.core import tensor2img
from ..registry import MODELS
from .basic_restorer import BasicRestorer
import time

@MODELS.register_module()
class MFQEv2Restorer(BasicRestorer):
    """MFQEv2 model for video quality enhancement.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_spynet_iter = train_cfg.get('fix_spynet_iter',
                                             0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet at the beginning
        if self.step_counter < self.fix_spynet_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_spynet_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    # def forward_test(self,
    #                  lq,
    #                  gt=None,
    #                  meta=None,
    #                  save_image=False,
    #                  save_path=None,
    #                  iteration=None):
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

        torch.cuda.synchronize()
        begin_time=time.time()
        output = self.generator(lq)
        torch.cuda.synchronize()
        end_time=time.time()
        print(end_time-begin_time)
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
