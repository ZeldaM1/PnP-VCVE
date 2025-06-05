# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import os
 
@DATASETS.register_module()
class BigMultipleGTMixCompressDataset(BaseSRDataset):
    """
    REDS_HR: dict(cprs15_folder=,cprs25_folder=,cprs35_folder,lq_folder,gt_folder)
    cprs15_folder,
                    cprs25_folder,
                    cprs35_folder,
                    lq_folder,
                    gt_folder,
                    num_input_frames,
                    pipeline,
                    scale,
                    val_partition='official',
                    repeat=1,
                    test_mode=False):

    """
    def __init__(self,
                 cprs15_folder,
                 cprs25_folder,
                 cprs35_folder,
                 lq_folder,
                 gt_folder,
                 num_input_frames,
                 pipeline,
                 scale,
                 val_partition='official',
                 repeat=1,
                 test_mode=False,
                 sequence_length={'HR':100,'LR':100,'DAVIS':20}):

        self.repeat = repeat
        if not isinstance(repeat, int):
            raise TypeError('"repeat" must be an integer, but got '
                            f'{type(repeat)}.')

        super().__init__(pipeline, scale, test_mode)

        self.cprs15_folder = dict(cprs15_folder)
        self.cprs25_folder = dict(cprs25_folder)
        self.cprs35_folder = dict(cprs35_folder)
        self.lq_folder = dict(lq_folder)
        self.gt_folder = dict(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.sequence_length=sequence_length
        self.data_infos = self.load_annotations()
        
  
 

    def load_annotations(self):
        """Load annotations for REDS dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # generate keys
        
        keys={}
        for dataname in self.lq_folder.keys():
            keys_dataset = [i.name for i in os.scandir(self.lq_folder[dataname])]
            keys[dataname]=keys_dataset
        
 
        data_infos = []
        for dataname in self.lq_folder.keys():
            for key in keys[dataname]:
                data_infos.append(
                    dict(
                        cprs15_path = self.cprs15_folder[dataname],
                        cprs25_path = self.cprs25_folder[dataname],
                        cprs35_path = self.cprs35_folder[dataname],
                        lq_path=self.lq_folder[dataname],
                        gt_path=self.gt_folder[dataname],
                        key=key,
                        sequence_length=self.sequence_length[dataname],  # REDS has 100 frames for each clip
                        num_input_frames=self.num_input_frames))

 
        
        return data_infos
