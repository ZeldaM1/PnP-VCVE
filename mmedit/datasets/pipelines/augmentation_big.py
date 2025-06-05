# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import numbers
import os
import os.path as osp
import random

import cv2
import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..registry import PIPELINES

@PIPELINES.register_module()
class Big_GenerateSegmentIndices_Mix_Compress:

    def __init__(self, interval_list, start_idx=0, filename_tmpl={'HR':'{:08d}.png','LR':'{:08d}.png','DAVIS':'{:05d}.png',}):
        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    def __call__(self, results):

        
        # key example: '000', 'calendar' (sequence name)
        clip_name = results['key']
        # print('---------------------',results['cprs15_path'],clip_name)
        # breakpoint()
        interval = np.random.choice(self.interval_list)

        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        start_frame_idx = np.random.randint(0, self.sequence_length - num_input_frames * interval + 1)
        end_frame_idx = start_frame_idx + num_input_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        neighbor_list = [v + self.start_idx for v in neighbor_list]

        # add the corresponding file paths
        cprs15_path_root = results['cprs15_path']
        cprs25_path_root = results['cprs25_path']
        cprs35_path_root = results['cprs35_path']
        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
         
        cprs15_path,cprs25_path,cprs35_path=[],[],[]

        for dataname in cprs15_path_root.keys():
            for v in neighbor_list:
                cprs15_path.append(osp.join(cprs15_path_root[dataname], clip_name, self.filename_tmpl[dataname].format(v)))
                cprs25_path.append(osp.join(cprs15_path_root[dataname], clip_name, self.filename_tmpl[dataname].format(v)))
                cprs35_path.append(osp.join(cprs15_path_root[dataname], clip_name, self.filename_tmpl[dataname].format(v)))
                lq_path.append(osp.join(cprs15_path_root[dataname], clip_name, self.filename_tmpl[dataname].format(v)))
                gt_path.append(osp.join(cprs15_path_root[dataname], clip_name, self.filename_tmpl[dataname].format(v)))
 
 
        results['cprs15_path'] = cprs15_path
        results['cprs25_path'] = cprs25_path
        results['cprs35_path'] = cprs35_path
        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        assert len(cprs15_path)== len(gt_path)
        assert len(cprs25_path)== len(gt_path)
        assert len(cprs35_path)== len(gt_path)
        assert len(lq_path)== len(gt_path)
 
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')
        return repr_str

 