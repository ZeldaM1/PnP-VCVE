# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import mmcv
import numpy as np
from mmcv.fileio import FileClient
import json
from mmedit.core.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox)
from ..registry import PIPELINES
import os

@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 random_compress=False,
                 compress_ratio=0.5,
                 load_qp_slice = False,
                 load_base_qp=False,
                 load_mv = False,
                 qp_slice_file=None,
                 dataset='reds',
                 use_bi_infor=False,
                 load_partition=False,
                 replace_qp_withIPB=False,
                 drconv=False,
                 **kwargs):

        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.convert_to = convert_to
        self.kwargs = kwargs
        self.file_client = None
        self.use_cache = use_cache
        self.cache = dict() if use_cache else None
        self.backend = backend
        self.random_compress = random_compress
        self.compress_ratio = compress_ratio
        self.load_qp_slice = load_qp_slice
        self.load_base_qp = load_base_qp
        self.qp_slice_dict=None
        self.dataset=dataset
        self.load_mv=load_mv
        self.use_bi_infor=use_bi_infor
        if (qp_slice_file is not None) and load_qp_slice:
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()
        self.last_is_B_frame=False
        self.load_partition=load_partition
        self.drconv=drconv
        # breakpoint()
        self.replace_qp_withIPB=replace_qp_withIPB
        

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
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

        if self.convert_to is not None:
            if self.channel_order == 'bgr' and self.convert_to.lower() == 'y':
                img = mmcv.bgr2ycbcr(img, y_only=True)
            elif self.channel_order == 'rgb':
                img = mmcv.rgb2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileList_Mix_Compress(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

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
        base_qp=0
        if self.key=='lq' and self.random_compress and (np.random.random() < self.compress_ratio):
            choose_crf = np.random.random()
            if choose_crf <0.33:
                filepaths = results['cprs15_path']
                base_qp=15
            elif (choose_crf > 0.33) and (choose_crf <0.66):
                filepaths = results['cprs25_path']
                base_qp=25
            elif (choose_crf >0.66):
                filepaths = results['cprs35_path']
                base_qp=35
        else:
            filepaths = results[f'{self.key}_path']

        # if self.key=='lq':
        #     print('!!!!!!!!!!1',base_qp,filepaths[3])

         
     
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]


        imgs = []
        shapes = []
        QPs =[]
        slices=[]
        mvs=[]

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
                # print('-----------',filepath,filepath_mv)
                 
                mv_backward_x_aft = np.zeros((h,w)).astype(np.float32)
                mv_backward_y_aft = np.zeros((h,w)).astype(np.float32)
                
                
                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: # forward mv
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0]=motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1]=motion_y # forward y
                    else: # reverse forward flow
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                        if self.use_bi_infor: #use bidirectional backward
                            mv_backward_x_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_x # forward x
                            mv_backward_y_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_y # forward y
                
                if self.last_is_B_frame and self.use_bi_infor:
                    mv[:,:,2:] = self.last_mv

                if mv_npy.shape[0]>0:
                    self.last_is_B_frame = not (np.min(mv_npy[:,0])==np.max(mv_npy[:,0]))
                    self.last_mv = np.stack([mv_backward_x_aft,mv_backward_y_aft],axis=2)
                mvs.append(mv)

                
            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                if self.dataset=='vimeo':
                    crf, dirname, subdirname, filename  = filepath_split[4], filepath_split[2], filepath_split[1],(filepath_split[0][2:].split('.')[0])
                    # print('---------------------',filepath_split)
                    # print('!!!!!!!!!!!!!!!!!', crf, dirname, subdirname, filename)
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][subdirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                else:
                    
                    crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)

        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices
        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs
        if self.load_base_qp and self.key=='lq':
            base_qps=[np.array(base_qp).reshape((1,1)) for n in range(len(imgs))]
            results['base_QPs'] = base_qps
        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results



@PIPELINES.register_module()
class LoadImageFromFileList_Mix_Compress_v2(LoadImageFromFile):

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
 
        if self.key=='lq' and self.random_compress and (np.random.random() < self.compress_ratio):
            choose_crf = np.random.random()
            if choose_crf <0.33:
                filepaths = results['cprs15_path']
            elif (choose_crf > 0.33) and (choose_crf <0.66):
                filepaths = results['cprs25_path']
            elif (choose_crf >0.66):
                filepaths = results['cprs35_path']
        else:
            filepaths = results[f'{self.key}_path']

         
     
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]


        imgs = []
        shapes = []
        QPs =[]
        slices=[]
        mvs=[]

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
                # print('-----------',filepath,filepath_mv)
                 
                mv_backward_x_aft = np.zeros((h,w)).astype(np.float32)
                mv_backward_y_aft = np.zeros((h,w)).astype(np.float32)
                
                
                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: # P frame + B frame forward
                        # reverse forward flow
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0]=motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1]=motion_y # forward y
                        # reverse forward flow
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                    else: # reverse forward flow
                        if self.use_bi_infor: #use bidirectional backward
                            mv_backward_x_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_x # forward x
                            mv_backward_y_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_y # forward y
                
                if self.last_is_B_frame and self.use_bi_infor:
                    mv[:,:,2:] = self.last_mv

                if mv_npy.shape[0]>0:
                    self.last_is_B_frame = not (np.min(mv_npy[:,0])==np.max(mv_npy[:,0]))
                    self.last_mv = np.stack([mv_backward_x_aft,mv_backward_y_aft],axis=2)
                mvs.append(mv)

                
            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                if self.dataset=='vimeo':
                    crf, dirname, subdirname, filename  = filepath_split[4], filepath_split[2], filepath_split[1],(filepath_split[0][2:].split('.')[0])
                    # print('---------------------',filepath_split)
                    # print('!!!!!!!!!!!!!!!!!', crf, dirname, subdirname, filename)
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][subdirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                else:
                    
                    crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)


        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices
        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results




@PIPELINES.register_module()
class LoadImageFromFileList_Mix_Compress_specify(LoadImageFromFile):
    def __init__(self,data_ratio=[0.25,0.5,0.75,1],**kwargs):
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
            elif (choose_crf >= self.x4_ratio) and (choose_crf < self.crf15_ratio):
                filepaths = results['cprs15_path']
            elif (choose_crf >= self.crf15_ratio) and (choose_crf < self.crf25_ratio):
                filepaths = results['cprs25_path']
            elif (choose_crf >= self.crf25_ratio) and (choose_crf < self.crf35_ratio):
                filepaths = results['cprs35_path']
        else:
            filepaths = results[f'{self.key}_path']

        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]


        imgs = []
        shapes = []
        QPs =[]
        slices=[]
        mvs=[]

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
                # print('-----------',filepath,filepath_mv)
                 
                mv_backward_x_aft = np.zeros((h,w)).astype(np.float32)
                mv_backward_y_aft = np.zeros((h,w)).astype(np.float32)
                
                
                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: # P frame + B frame forward
                        # reverse forward flow
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0]=motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1]=motion_y # forward y
                        # reverse forward flow
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                    else: # reverse forward flow
                        if self.use_bi_infor: #use bidirectional backward
                            mv_backward_x_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_x # forward x
                            mv_backward_y_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_y # forward y
                
                if self.last_is_B_frame and self.use_bi_infor:
                    mv[:,:,2:] = self.last_mv

                if mv_npy.shape[0]>0:
                    self.last_is_B_frame = not (np.min(mv_npy[:,0])==np.max(mv_npy[:,0]))
                    self.last_mv = np.stack([mv_backward_x_aft,mv_backward_y_aft],axis=2)
                mvs.append(mv)

                
            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                if self.dataset=='vimeo':
                    crf, dirname, subdirname, filename  = filepath_split[4], filepath_split[2], filepath_split[1],(filepath_split[0][2:].split('.')[0])
                    # print('---------------------',filepath_split)
                    # print('!!!!!!!!!!!!!!!!!', crf, dirname, subdirname, filename)
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][subdirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                else:
                    
                    crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))
                    if crf.startswith('crf'):
                        qp_slice = self.qp_slice_dict[crf][dirname][filename]
                        qp = qp_slice['QP']
                        slice = qp_slice['slice']
                    else:
                        qp=0.0
                        slice = 'I' if filename=='0' else 'P'
                

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)


        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices
        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results







@PIPELINES.register_module()
class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

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
 
        # if self.key=='lq' and self.random_compress and (np.random.random() < self.compress_ratio):
        #     filepaths = results['cprs_path']
        # else:
        filepaths = results[f'{self.key}_path']
        # print("!!!!!!!!!!!!!!!!!", filepaths[3:5])
        if self.key=='lq':
            base_qp=0
            if 'crf' in filepaths[0]:
                base_qp=int(filepaths[0].split('crf')[1].split('/')[0])     
     
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        QPs=[]
        slices=[]
        mvs=[]

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
                # print('-----------',filepath,filepath_mv)
                 
                mv_backward_x_aft = np.zeros((h,w)).astype(np.float32)
                mv_backward_y_aft = np.zeros((h,w)).astype(np.float32)
                
                
                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: # forward mv
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0]=motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1]=motion_y # forward y
                    else: # reverse forward flow
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                        if self.use_bi_infor: #use bidirectional backward
                            mv_backward_x_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_x # forward x
                            mv_backward_y_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_y # forward y
                
                if self.last_is_B_frame and self.use_bi_infor:
                    mv[:,:,2:] = self.last_mv

                if mv_npy.shape[0]>0:
                    self.last_is_B_frame = not (np.min(mv_npy[:,0])==np.max(mv_npy[:,0]))
                    self.last_mv = np.stack([mv_backward_x_aft,mv_backward_y_aft],axis=2)
                mvs.append(mv)


            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))

                if crf.startswith('crf'):
                    qp_slice = self.qp_slice_dict[crf][dirname][filename]
                    qp = qp_slice['QP']
                    slice = qp_slice['slice']
                else:
                    qp=0.0
                    slice = 'I' if filename=='0' else 'P'

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)

        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices

        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs

        if self.load_base_qp and self.key=='lq':
            base_qps=[np.array(base_qp).reshape((1,1)) for n in range(len(imgs))]
            results['base_QPs'] = base_qps

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results



@PIPELINES.register_module()
class LoadImageFromFileList_v2(LoadImageFromFile):
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
 
        # if self.key=='lq' and self.random_compress and (np.random.random() < self.compress_ratio):
        #     filepaths = results['cprs_path']
        # else:
        filepaths = results[f'{self.key}_path']
        # print("!!!!!!!!!!!!!!!!!", filepaths[3:5])
     
     
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        QPs=[]
        slices=[]
        mvs=[]

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
                # print('-----------',filepath,filepath_mv)
                 
                mv_backward_x_aft = np.zeros((h,w)).astype(np.float32)
                mv_backward_y_aft = np.zeros((h,w)).astype(np.float32)
                
                
                for idx in range(mv_npy.shape[0]):
                    direction, w,h, x_w, y_w, x, y, motion_x, motion_y, scale =  mv_npy[idx]
                    x,y,w,h,x_w, y_w=int(x),int(y),int(w),int(h),int(x_w), int(y_w)
                    motion_x=motion_x/scale
                    motion_y=motion_y/scale
                    if direction<0: # P frame + B frame forward
                        # reverse forward flow
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,0]=motion_x # forward x
                        mv[y - h // 2:y + h // 2, x - w // 2:x + w // 2,1]=motion_y # forward y
                        # reverse forward flow
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,2]= - motion_x # backward x
                        mv[y_w - h // 2:y_w + h // 2, x_w - w // 2:x_w + w // 2,3]= - motion_y # backward y
                    else: # reverse forward flow
                        if self.use_bi_infor: #use bidirectional backward
                            mv_backward_x_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_x # forward x
                            mv_backward_y_aft[y - h // 2:y + h // 2, x - w // 2:x + w // 2]=motion_y # forward y
                
                if self.last_is_B_frame and self.use_bi_infor:
                    mv[:,:,2:] = self.last_mv

                if mv_npy.shape[0]>0:
                    self.last_is_B_frame = not (np.min(mv_npy[:,0])==np.max(mv_npy[:,0]))
                    self.last_mv = np.stack([mv_backward_x_aft,mv_backward_y_aft],axis=2)
                mvs.append(mv)


            if self.load_qp_slice and self.key=='lq':
                filepath_split = filepath.split('/')[::-1]
                crf, dirname, filename = filepath_split[3], filepath_split[1], str(int(filepath_split[0].split('.')[0]))

                if crf.startswith('crf'):
                    qp_slice = self.qp_slice_dict[crf][dirname][filename]
                    qp = qp_slice['QP']
                    slice = qp_slice['slice']
                else:
                    qp=0.0
                    slice = 'I' if filename=='0' else 'P'

                qp = np.array(qp).reshape((1,1))
                qp = np.expand_dims(qp, axis=2)
                QPs.append(qp)

                slice = np.array(ord(slice)).reshape((1,1))
                slice = np.expand_dims(slice, axis=2)
                slices.append(slice)

        if self.load_qp_slice and self.key=='lq':
            assert len(imgs) == len(slices)
            assert len(imgs) == len(QPs)
            results['QPs'] = QPs
            results['slices'] = slices

        if self.load_mv and self.key=='lq':
            assert len(imgs) == len(mvs)
            results['mvs'] = mvs

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results

 

@PIPELINES.register_module()
class RandomLoadResizeBg:
    """Randomly load a background image and resize it.

    Required key is "fg", added key is "bg".

    Args:
        bg_dir (str): Path of directory to load background images from.
        io_backend (str): io backend where images are store. Default: 'disk'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 bg_dir,
                 io_backend='disk',
                 flag='color',
                 channel_order='bgr',
                 **kwargs):
        self.bg_dir = bg_dir
        self.bg_list = list(mmcv.scandir(bg_dir))
        self.io_backend = io_backend
        self.flag = flag
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.file_client = None

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
        h, w = results['fg'].shape[:2]
        idx = np.random.randint(len(self.bg_list))
        filepath = Path(self.bg_dir).joinpath(self.bg_list[idx])
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        bg = mmcv.imresize(img, (w, h), interpolation='bicubic')
        results['bg'] = bg
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(bg_dir='{self.bg_dir}')"


@PIPELINES.register_module()
class LoadMask:
    """Load Mask for multiple types.

    For different types of mask, users need to provide the corresponding
    config dict.

    Example config for bbox:

    .. code-block:: python

        config = dict(img_shape=(256, 256), max_bbox_shape=128)

    Example config for irregular:

    .. code-block:: python

        config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            max_angle=4.,
            length_range=(10, 100),
            brush_width=(10, 40),
            area_ratio_range=(0.15, 0.5))

    Example config for ff:

    .. code-block:: python

        config = dict(
            img_shape=(256, 256),
            num_vertices=(4, 12),
            mean_angle=1.2,
            angle_range=0.4,
            brush_width=(12, 40))

    Example config for set:

    .. code-block:: python

        config = dict(
            mask_list_file='xxx/xxx/ooxx.txt',
            prefix='/xxx/xxx/ooxx/',
            io_backend='disk',
            flag='unchanged',
            file_client_kwargs=dict()
        )

        The mask_list_file contains the list of mask file name like this:
            test1.jpeg
            test2.jpeg
            ...
            ...

        The prefix gives the data path.

    Args:
        mask_mode (str): Mask mode in ['bbox', 'irregular', 'ff', 'set',
            'file'].
            * bbox: square bounding box masks.
            * irregular: irregular holes.
            * ff: free-form holes from DeepFillv2.
            * set: randomly get a mask from a mask set.
            * file: get mask from 'mask_path' in results.
        mask_config (dict): Params for creating masks. Each type of mask needs
            different configs.
    """

    def __init__(self, mask_mode='bbox', mask_config=None):
        self.mask_mode = mask_mode
        self.mask_config = dict() if mask_config is None else mask_config
        assert isinstance(self.mask_config, dict)

        # set init info if needed in some modes
        self._init_info()

    def _init_info(self):
        if self.mask_mode == 'set':
            # get mask list information
            self.mask_list = []
            mask_list_file = self.mask_config['mask_list_file']
            with open(mask_list_file, 'r') as f:
                for line in f:
                    line_split = line.strip().split(' ')
                    mask_name = line_split[0]
                    self.mask_list.append(
                        Path(self.mask_config['prefix']).joinpath(mask_name))
            self.mask_set_size = len(self.mask_list)
            self.io_backend = self.mask_config['io_backend']
            self.flag = self.mask_config['flag']
            self.file_client_kwargs = self.mask_config['file_client_kwargs']
            self.file_client = None
        elif self.mask_mode == 'file':
            self.io_backend = 'disk'
            self.flag = 'unchanged'
            self.file_client_kwargs = dict()
            self.file_client = None

    def _get_random_mask_from_set(self):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend,
                                          **self.file_client_kwargs)
        # minus 1 to avoid out of range error
        mask_idx = np.random.randint(0, self.mask_set_size)
        mask_bytes = self.file_client.get(self.mask_list[mask_idx])
        mask = mmcv.imfrombytes(mask_bytes, flag=self.flag)  # HWC, BGR
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        else:
            mask = mask[:, :, 0:1]

        mask[mask > 0] = 1.
        return mask

    def _get_mask_from_file(self, path):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend,
                                          **self.file_client_kwargs)
        mask_bytes = self.file_client.get(path)
        mask = mmcv.imfrombytes(mask_bytes, flag=self.flag)  # HWC, BGR
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        else:
            mask = mask[:, :, 0:1]

        mask[mask > 0] = 1.
        return mask

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.mask_mode == 'bbox':
            mask_bbox = random_bbox(**self.mask_config)
            mask = bbox2mask(self.mask_config['img_shape'], mask_bbox)
            results['mask_bbox'] = mask_bbox
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(**self.mask_config)
        elif self.mask_mode == 'set':
            mask = self._get_random_mask_from_set()
        elif self.mask_mode == 'ff':
            mask = brush_stroke_mask(**self.mask_config)
        elif self.mask_mode == 'file':
            mask = self._get_mask_from_file(results['mask_path'])
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        results['mask'] = mask
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(mask_mode='{self.mask_mode}')"


@PIPELINES.register_module()
class GetSpatialDiscountMask:
    """Get spatial discounting mask constant.

    Spatial discounting mask is first introduced in:
    Generative Image Inpainting with Contextual Attention.

    Args:
        gamma (float, optional): Gamma for computing spatial discounting.
            Defaults to 0.99.
        beta (float, optional): Beta for computing spatial discounting.
            Defaults to 1.5.
    """

    def __init__(self, gamma=0.99, beta=1.5):
        self.gamma = gamma
        self.beta = beta

    def spatial_discount_mask(self, mask_width, mask_height):
        """Generate spatial discounting mask constant.

        Args:
            mask_width (int): The width of bbox hole.
            mask_height (int): The height of bbox height.

        Returns:
            np.ndarray: Spatial discounting mask.
        """
        w, h = np.meshgrid(np.arange(mask_width), np.arange(mask_height))
        grid_stack = np.stack([h, w], axis=2)
        mask_values = (self.gamma**(np.minimum(
            grid_stack, [mask_height - 1, mask_width - 1] - grid_stack) *
                                    self.beta)).max(
                                        axis=2, keepdims=True)

        return mask_values

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        mask_bbox = results['mask_bbox']
        mask = results['mask']
        mask_height, mask_width = mask_bbox[-2:]
        discount_hole = self.spatial_discount_mask(mask_width, mask_height)
        discount_mask = np.zeros_like(mask)
        discount_mask[mask_bbox[0]:mask_bbox[0] + mask_height,
                      mask_bbox[1]:mask_bbox[1] + mask_width,
                      ...] = discount_hole

        results['discount_mask'] = discount_mask

        return results

    def __repr__(self):
        return self.__class__.__name__ + (f'(gamma={self.gamma}, '
                                          f'beta={self.beta})')


@PIPELINES.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load a pair of images from file.

    Each sample contains a pair of images, which are concatenated in the w
    dimension (a|b). This is a special loading class for generation paired
    dataset. It loads a pair of images as the common loader does and crops
    it into two images with the same shape in different domains.

    Required key is "pair_path". Added or modified keys are "pair",
    "pair_ori_shape", "ori_pair", "img_a", "img_b", "img_a_path",
    "img_b_path", "img_a_ori_shape", "img_b_ori_shape", "ori_img_a" and
    "ori_img_b".

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

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
        filepath = str(results[f'{self.key}_path'])
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        # crop pair into a and b
        w = img.shape[1]
        if w % 2 != 0:
            raise ValueError(
                f'The width of image pair must be even number, but got {w}.')
        new_w = w // 2
        img_a = img[:, :new_w, :]
        img_b = img[:, new_w:, :]

        results['img_a'] = img_a
        results['img_b'] = img_b
        results['img_a_path'] = filepath
        results['img_b_path'] = filepath
        results['img_a_ori_shape'] = img_a.shape
        results['img_b_ori_shape'] = img_b.shape
        if self.save_original_img:
            results['ori_img_a'] = img_a.copy()
            results['ori_img_b'] = img_b.copy()

        return results
