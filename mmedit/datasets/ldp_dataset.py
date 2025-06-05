# Copyright (c) ryanxingql. All rights reserved.
import os.path as osp
import random
from glob import glob
import os
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import json

@DATASETS.register_module()
class LDPPQFDataset(BaseSRDataset):
    def __init__(
        self,
        cprs15_folder,
        cprs25_folder,
        cprs35_folder,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.cprs15_folder=str(cprs15_folder)
        self.cprs25_folder=str(cprs25_folder)
        self.cprs35_folder=str(cprs35_folder)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        self.imgs_num={}


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()


        keys = []
        for fin in os.scandir(lq_folder):
            # breakpoint()
            img_list=sorted(glob(f'{lq_folder}/{fin.name}/*.png'))#.replace(f'{lq_folder}/','')
            keys.extend(img_list)
            self.imgs_num[fin.name] = (len(img_list))
        self.keys=keys

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
        find_flag=False
        left_idx=int(filename)-1
        while(left_idx>=0):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                left_idx-=1
        if not find_flag:
            left_idx=int(filename)

        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=(self.imgs_num[dirname]-1)):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                right_idx+=1
        if not find_flag:
            right_idx=int(filename)

        return left_idx,right_idx

    def load_annotations(self):
 
        keys = []
        for fin in os.scandir(self.cprs15_folder):
            img_list=sorted(glob(f'{self.cprs15_folder}/{fin.name}/*.png'))
            keys.extend(img_list)

        data_infos = []
        for key in keys:
            clip_img=key.replace(f'{self.cprs15_folder}/','')
            clip,filename=clip_img.split('/')
            # breakpoint()
            filename= int(filename.split('.png')[0])
            if (self.qp_slice_dict['crf15'][clip][str(filename)]['slice']=='I') or (self.qp_slice_dict['crf15'][clip][str(filename)]['slice']=='P'):
                # breakpoint()
                left_pqf_idx15,right_pqf_idx15=self.find_left_right_pqf('crf15',clip,filename)
                left_pqf_idx25,right_pqf_idx25=self.find_left_right_pqf('crf25',clip,filename)
                left_pqf_idx35,right_pqf_idx35=self.find_left_right_pqf('crf35',clip,filename)
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf('x4',clip,filename)
                # print('-------------',clip_img)
                data_infos.append(
                    dict(
                        cprs15_path = [self.cprs15_folder, left_pqf_idx15, right_pqf_idx15],
                        cprs25_path = [self.cprs25_folder, left_pqf_idx25, right_pqf_idx25],
                        cprs35_path = [self.cprs35_folder, left_pqf_idx35,right_pqf_idx35],
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos



@DATASETS.register_module()
class LDPNonPQFDataset(BaseSRDataset):
    def __init__(
        self,
        cprs15_folder,
        cprs25_folder,
        cprs35_folder,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.cprs15_folder=str(cprs15_folder)
        self.cprs25_folder=str(cprs25_folder)
        self.cprs35_folder=str(cprs35_folder)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        self.imgs_num={}


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()


        keys = []
        for fin in os.scandir(lq_folder):
            # breakpoint()
            img_list=sorted(glob(f'{lq_folder}/{fin.name}/*.png'))#.replace(f'{lq_folder}/','')
            keys.extend(img_list)
            self.imgs_num[fin.name] = (len(img_list))
        self.keys=keys

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
 
        left_idx=int(filename)-1
        while(left_idx>=0):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                break
            else:
                left_idx-=1

        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=(self.imgs_num[dirname]-1)):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                break
            else:
                right_idx+=1

        return left_idx,right_idx

    def load_annotations(self):
 
        keys = []
        for fin in os.scandir(self.cprs15_folder):
            img_list=sorted(glob(f'{self.cprs15_folder}/{fin.name}/*.png'))
            keys.extend(img_list)

        data_infos = []
        for key in keys:
            clip_img=key.replace(f'{self.cprs15_folder}/','')
            clip,filename=clip_img.split('/')
            # breakpoint()
            filename= int(filename.split('.png')[0])
            if self.qp_slice_dict['crf15'][clip][str(filename)]['slice']=='B':
                # breakpoint()
                left_pqf_idx15,right_pqf_idx15=self.find_left_right_pqf('crf15',clip,filename)
                left_pqf_idx25,right_pqf_idx25=self.find_left_right_pqf('crf25',clip,filename)
                left_pqf_idx35,right_pqf_idx35=self.find_left_right_pqf('crf35',clip,filename)
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf('x4',clip,filename)
                # print('-------------',clip_img)
                data_infos.append(
                    dict(
                        cprs15_path = [self.cprs15_folder, left_pqf_idx15, right_pqf_idx15],
                        cprs25_path = [self.cprs25_folder, left_pqf_idx25, right_pqf_idx25],
                        cprs35_path = [self.cprs35_folder, left_pqf_idx35,right_pqf_idx35],
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos


@DATASETS.register_module()
class LDPNonPQFDataset_test(BaseSRDataset):
    def __init__(
        self,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        self.imgs_num={}


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()


        keys = []
        for fin in os.scandir(lq_folder):
            # breakpoint()
            img_list=sorted(glob(f'{lq_folder}/{fin.name}/*.png'))#.replace(f'{lq_folder}/','')
            keys.extend(img_list)
            self.imgs_num[fin.name] = (len(img_list))
        self.keys=keys
        # breakpoint()

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
 
        left_idx=int(filename)-1
        while(left_idx>=0):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                break
            else:
                left_idx-=1

        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=(self.imgs_num[dirname]-1)):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                break
            else:
                right_idx+=1

        return left_idx,right_idx

    def load_annotations(self):
 
        keys = []
        for fin in os.scandir(self.lq_folder):
            img_list=sorted(glob(f'{self.lq_folder}/{fin.name}/*.png'))
            keys.extend(img_list)

        data_infos = []
        for key in keys:
            clip_img=key.replace(f'{self.lq_folder}/','')
            clip,filename=clip_img.split('/')
            filename= int(filename.split('.png')[0])
            crf=self.lq_folder.split('/')[::-1][1]
            if self.qp_slice_dict[crf][clip][str(filename)]['slice']=='B' or ('crf' not in crf):
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf(crf,clip,filename)
                # print(self.qp_slice_dict[crf][clip][str(filename)]['slice'], clip_img)
                data_infos.append(
                    dict(
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos




@DATASETS.register_module()
class LDPNonPQFDataset_test_kitti(BaseSRDataset):
    def __init__(
        self,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        self.imgs_num={}
        all_imgs=sorted(glob(f'{lq_folder}/*.png'))
        self.keys=all_imgs


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()
 
        for fin in all_imgs:
            self.imgs_num[os.path.basename(fin).split('_')[0]] = 2

 

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
        # breakpoint()
 
        left_idx=int(filename)-1
        while(left_idx>=0):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                break
            else:
                left_idx-=1

        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=(self.imgs_num[dirname]-1)):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                break
            else:
                right_idx+=1

        return left_idx,right_idx

    def load_annotations(self):
 
        data_infos = []
        for key in self.keys:
            # breakpoint()
            clip_img=key.replace(f'{self.lq_folder}/','')
            clip,filename=clip_img.split('_')
            filename= int(filename.split('.png')[0])
            crf=self.lq_folder.split('/')[::-1][1].split('_')[2]
            if self.qp_slice_dict[crf][clip][str(filename)]['slice']=='B' or ('crf' not in crf):
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf(crf,clip,filename)
                # print(self.qp_slice_dict[crf][clip][str(filename)]['slice'], clip_img)
                data_infos.append(
                    dict(
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos


@DATASETS.register_module()
class LDPPQFDataset_test(BaseSRDataset):
    def __init__(
        self,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        self.imgs_num={}


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()


        keys = []
        for fin in os.scandir(lq_folder):
            # breakpoint()
            img_list=sorted(glob(f'{lq_folder}/{fin.name}/*.png'))#.replace(f'{lq_folder}/','')
            keys.extend(img_list)
            self.imgs_num[fin.name] = (len(img_list))
        self.keys=keys

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
        find_flag=False
        left_idx=int(filename)-1
        while(left_idx>=0):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                left_idx-=1
        if not find_flag:
            left_idx=int(filename)
            
        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=(self.imgs_num[dirname]-1)):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                right_idx+=1
        if not find_flag:
            right_idx=int(filename)

        return left_idx,right_idx

    def load_annotations(self):
 
        keys = []
        for fin in os.scandir(self.lq_folder):
            img_list=sorted(glob(f'{self.lq_folder}/{fin.name}/*.png'))
            keys.extend(img_list)

        data_infos = []
        for key in keys:
            clip_img=key.replace(f'{self.lq_folder}/','')
            clip,filename=clip_img.split('/')
            filename= int(filename.split('.png')[0])
            crf=self.lq_folder.split('/')[::-1][1]
            if (self.qp_slice_dict[crf][clip][str(filename)]['slice']=='I') or (self.qp_slice_dict[crf][clip][str(filename)]['slice']=='P') or ('crf' not in crf):
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf(crf,clip,filename)
                # print(self.qp_slice_dict[crf][clip][str(filename)]['slice'], clip_img)
                data_infos.append(
                    dict(
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos



@DATASETS.register_module()
class LDPPQFDataset_test_kitti(BaseSRDataset):
    def __init__(
        self,
        lq_folder,
        gt_folder,
        pipeline,
        scale,
        qp_slice_file,
        filename_tmpl='f{:03d}',
        i_frame_idx=0,
        num_input_frames=1,
        test_mode=False,
    ):
        super().__init__(
            pipeline,
            scale,
        )
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.i_frame_idx = i_frame_idx

        self.filename_tmpl = filename_tmpl

        self.test_mode = test_mode
        self.num_input_frames = num_input_frames
        all_imgs=sorted(glob(f'{lq_folder}/*.png'))
        self.keys=all_imgs


        if (qp_slice_file is not None):
            with open(qp_slice_file, 'r') as fr:
                self.qp_slice_dict = json.load(fr)
                fr.close()
        # breakpoint()
        self.imgs_num={}
        for fin in all_imgs:
            self.imgs_num[os.path.basename(fin).split('_')[0]] = 2
     

        self.data_infos = self.load_annotations()

    def find_left_right_pqf(self, crf,dirname,filename):
        left_idx,right_idx=int(filename),int(filename)
        if not 'crf' in crf: 
            if filename>0:
                left_idx=filename-1
            if filename<self.imgs_num[dirname]-1:
                right_idx=filename+1
            return left_idx,right_idx
        #---------------- crf situation
        PQF_type=['I','P']
        find_flag=False
        left_idx=int(filename)-1
        while(left_idx>=10):
            if self.qp_slice_dict[crf][dirname][str(left_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                left_idx-=1
        if not find_flag:
            left_idx=int(filename)
            
        find_flag=False
        right_idx=int(filename)+1
        while(right_idx<=11):
            if self.qp_slice_dict[crf][dirname][str(right_idx)]['slice'] in PQF_type:
                find_flag=True
                break
            else:
                right_idx+=1
        if not find_flag:
            right_idx=int(filename)

        return left_idx,right_idx

    def load_annotations(self):
 
        
        data_infos = []
         
        for key in self.keys:
   
            clip_img=key.replace(f'{self.lq_folder}/','')
            clip,filename=clip_img.split('_')
            filename= int(filename.split('.png')[0])
            crf=self.lq_folder.split('/')[::-1][1].split('_')[2] 
            # breakpoint()
      
            if (self.qp_slice_dict[crf][clip][str(filename)]['slice']=='I') or (self.qp_slice_dict[crf][clip][str(filename)]['slice']=='P') or ('crf' not in crf):
                left_pqf_idxx4,right_pqf_idxx4=self.find_left_right_pqf(crf,clip,filename)
                # if filename==10:
                #     left_pqf_idxx4,right_pqf_idxx4=11,11
                # if filename==11:
                #     left_pqf_idxx4,right_pqf_idxx4=10,10
                # breakpoint()
                # print(self.qp_slice_dict[crf][clip][str(filename)]['slice'], clip_img)
                data_infos.append(
                    dict(
                        lq_path=[self.lq_folder, left_pqf_idxx4, right_pqf_idxx4],
                        gt_path=self.gt_folder,
                        key=clip_img,
                        sequence_length=self.imgs_num[clip],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))
 
    
        return data_infos


 