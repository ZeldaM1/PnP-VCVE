# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import numpy as np
from mmcv.fileio import FileClient
import json
from mmedit.core.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox)
from ..registry import PIPELINES
import os
from .loading import LoadImageFromFile
 
 
@PIPELINES.register_module()
class LoadImageFromFileList_Mix_NonPQF(LoadImageFromFile):
    def __init__(self, data_ratio=[0.25,0.5,0.75,1],**kwargs):
        super().__init__(**kwargs)
        self.x4_ratio=data_ratio[0]
        self.crf15_ratio=data_ratio[1]
        self.crf25_ratio=data_ratio[2]
        self.crf35_ratio=data_ratio[3]

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        if self.key=='lq' and self.random_compress:
            choose_crf = np.random.random()  
            if choose_crf <self.x4_ratio:
                filepaths = results['lq_path']
                base_qp=0
            elif (choose_crf >= self.x4_ratio) and (choose_crf < self.crf15_ratio):
                filepaths = results['cprs15_path']
                base_qp=15
            elif (choose_crf >= self.crf15_ratio) and (choose_crf < self.crf25_ratio):
                filepaths = results['cprs25_path']
                base_qp=25
            elif (choose_crf >= self.crf25_ratio) and (choose_crf < self.crf35_ratio):
                filepaths = results['cprs35_path']
                base_qp=35
        else:
            filepaths = results[f'{self.key}_path']

         
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]
        

        p_offset=0
        imgs = []
        shapes = []
 

        if self.save_original_img:
            ori_imgs = []
        
        for filepath in filepaths:
            if self.use_cache:
                if filepath in self.cache:
                    img = self.cache[filepath]
                else:
                    img_bytes = self.file_client.get(filepath)
                    img = mmcv.imfrombytes(
                        img_bytes,
                        flag=self.flag,
                        channel_order=self.channel_order,
                        backend=self.backend)  # HWC
                    self.cache[filepath] = img
            else:
                img_bytes = self.file_client.get(filepath)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.flag,
                    channel_order=self.channel_order,
                    backend=self.backend)  # HWC

            # convert to y-channel, if specified
            if self.convert_to is not None:
                if self.channel_order == 'bgr' and self.convert_to.lower(
                ) == 'y':
                    img = mmcv.bgr2ycbcr(img, y_only=True)
                elif self.channel_order == 'rgb':
                    img = mmcv.rgb2ycbcr(img, y_only=True)
                else:
                    raise ValueError('Currently support only "bgr2ycbcr" or '
                                     '"bgr2ycbcr".')

            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            imgs.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())
 
        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        # breakpoint()

        return results


  
 