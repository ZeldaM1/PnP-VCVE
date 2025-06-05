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
class LoadImageFromFileList_Mix_Compress_ipb(LoadImageFromFile):
    def __init__(self, data_ratio=[0.25,0.5,0.75,1],replace_qp_withIPB=False, **kwargs):
        super().__init__(**kwargs)
        self.x4_ratio=data_ratio[0]
        self.crf15_ratio=data_ratio[1]
        self.crf25_ratio=data_ratio[2]
        self.crf35_ratio=data_ratio[3]
        self.replace_qp_withIPB=replace_qp_withIPB

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
        QPs =[]
        slices=[]
        mvs=[]
        partitions=[]

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

            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                if self.dataset=='vimeo':
                    crf, dirname, subdirname, filename  = filepath_split[4], filepath_split[2], filepath_split[1],(filepath_split[0][2:].split('.')[0])
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][subdirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                else:
                    # print('##########################',filepath)
                    # print('_--------------------',filepath_split)
                    crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][filename]
                        slice = qp_slice['slice']
                        qp = qp_slice['QP'] if (not self.replace_qp_withIPB) else ord(slice)
                    else:
                        slice = 'I' if filename=='0' else 'P'
                        qp=0.0 if (not self.replace_qp_withIPB) else ord(slice) 
                

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                is_B_frame = (slice=='B')
                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)


            
            if self.load_mv and self.key=='lq':
                if self.dataset=='vimeo':
                    filepath_mv_dir, filepath_mv_idx=filepath.split('/im')
                    filepath_mv_idx=int(filepath_mv_idx.split('.png')[0])-1
                    filepath_mv_dir = filepath_mv_dir.replace('png','mv')
                    filepath_mv=os.path.join(filepath_mv_dir,'{:08d}.npy'.format(filepath_mv_idx))   
                else:
                    filepath_mv=filepath.replace('.png','.npy').replace('png','mv')
                             
                mv_npy=np.load(filepath_mv).astype(np.float32)               
                h,w,_=img.shape #(64, 112, 3) uint8
                mv = np.zeros((h,w,4)).astype(np.float32)  
                if self.load_partition:
                    if self.drconv:
                        partition = np.zeros((h,w,3)).astype(np.float32)
                        partition_ch={'256': 0, '128': 1, '64': 2}
                    else:
                        partition = np.zeros((h,w,1)).astype(np.float32)  

                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: 
                        # B frame forward
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0] = motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1] = motion_y # forward y
                    elif direction > 0 and is_B_frame:
                        # B frame backward
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,2] = motion_x # backward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,3] = motion_y # backward y
                    elif direction > 0 and (not is_B_frame): # P frame, reverse 
                        # reverse forward flow
                        mvs[-p_offset][y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mvs[-p_offset][y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                    else:
                        assert TypeError("frame type do not exist")

                    if self.load_partition:
                        if self.drconv:
                            partition[y - h // 2:y + h // 2, x - w // 2:x + w // 2,partition_ch[str(w*h)]]=1
                        else:
                            partition[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=255/(w * h)*64

                if self.load_partition:
                    partitions.append(partition) 
                mvs.append(mv)
                p_offset = p_offset + 1 if is_B_frame else 1



        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs

        if self.load_partition and self.key=='lq':
            assert len(imgs) == len(partitions)
            results['partitions'] = partitions

        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices
        
        if self.load_base_qp and self.key=='lq':
            base_qps=[np.array(base_qp).reshape((1,1)) for n in range(len(imgs))]
            results['base_QPs'] = base_qps

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        # breakpoint()

        return results


  


@PIPELINES.register_module()
class LoadImageFromFileList_ipb(LoadImageFromFile):
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        filepaths = results[f'{self.key}_path']
        if self.key=='lq':
            base_qp=int(filepaths[0].split('crf')[1].split('/')[0]) if ('crf' in filepaths[0]) else 0
     
     
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        QPs=[]
        slices=[]
        mvs=[]
        partitions=[]

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

            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))

                if crf.startswith('crf'):
                    qp_slice = self.qp_slice_dict[crf][dirname][filename]
                    slice = qp_slice['slice']
                    qp = qp_slice['QP'] if (not self.replace_qp_withIPB) else ord(slice)
                else:
                    slice = 'I' if filename=='0' else 'P'
                    qp=0.0 if (not self.replace_qp_withIPB) else ord(slice) 

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)
                is_B_frame = (slice=='B')
                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)
            

            if self.load_mv and self.key=='lq':
                if self.dataset=='vimeo':
                    filepath_mv_dir, filepath_mv_idx=filepath.split('/im')
                    filepath_mv_idx=int(filepath_mv_idx.split('.png')[0])-1
                    filepath_mv_dir = filepath_mv_dir.replace('png','mv')
                    filepath_mv=os.path.join(filepath_mv_dir,'{:08d}.npy'.format(filepath_mv_idx))   
                else:
                    filepath_mv=filepath.replace('.png','.npy').replace('png','mv')
               
                mv_npy=np.load(filepath_mv).astype(np.float32)               
                h,w,_=img.shape #(64, 112, 3) uint8
                mv = np.zeros((h,w,4)).astype(np.float32) # mv_forward(x,y) + mv_backward(x,y)
                if self.load_partition:
                    if self.drconv:
                        partition = np.zeros((h,w,3)).astype(np.float32)
                        partition_ch={'256': 0, '128': 1, '64': 2}
                    else:
                        partition = np.zeros((h,w,1)).astype(np.float32)  

                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: 
                        # B frame forward
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0] = motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1] = motion_y # forward y
                    elif direction > 0 and is_B_frame:
                        # B frame backward
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,2] = motion_x # backward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,3] = motion_y # backward y
                    elif direction > 0 and (not is_B_frame): # P frame, reverse 
                        # reverse forward flow
                        # print('+++++++++++++',p_offset,is_B_frame, slice)
                        mvs[-p_offset][y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mvs[-p_offset][y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                    else:
                        assert TypeError("frame type do not exist")

                    if self.load_partition:
                        if self.drconv:
                            partition[y - h // 2:y + h // 2, x - w // 2:x + w // 2,partition_ch[str(w*h)]]=1
                        else:
                            partition[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=255/(w * h)*64

                if self.load_partition:
                    partitions.append(partition) 

                mvs.append(mv)
                p_offset = p_offset + 1 if is_B_frame else 1



        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs
        
        if self.load_partition and self.key=='lq':
            assert len(imgs) == len(partitions)
            results['partitions'] = partitions
        
        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices
        
        if self.load_base_qp and self.key=='lq':
            base_qps=[np.array(base_qp).reshape((1,1)) for n in range(len(imgs))]
            results['base_QPs'] = base_qps

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results

  