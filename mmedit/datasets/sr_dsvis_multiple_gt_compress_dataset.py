# Copyright (c) OpenMMLab. All rights reserved.
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
import os
import glob
 
@DATASETS.register_module()
class DAVISMultipleGTMixCompressDataset(BaseSRDataset):
    """REDS dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        repeat (int): Number of replication of the validation set. This is used
            to allow training REDS4 with more than 4 GPUs. For example, if
            8 GPUs are used, this number can be set to 2. Default: 1.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
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
                 test_mode=False):

        self.repeat = repeat
        if not isinstance(repeat, int):
            raise TypeError('"repeat" must be an integer, but got '
                            f'{type(repeat)}.')

        super().__init__(pipeline, scale, test_mode)

        self.cprs15_folder = str(cprs15_folder)
        self.cprs25_folder = str(cprs25_folder)
        self.cprs35_folder = str(cprs35_folder)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.val_partition = val_partition
        self.data_infos = self.load_annotations()
        
        
        

  
    def load_annotations(self):
        """Load annotations for REDS dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # generate keys
        keys = [i.name for i in os.scandir(self.lq_folder)]
 
        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    cprs15_path = self.cprs15_folder,
                    cprs25_path = self.cprs25_folder,
                    cprs35_path = self.cprs35_folder,
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=20,  # REDS has 100 frames for each clip
                    num_input_frames=self.num_input_frames))

 
        
        return data_infos

@DATASETS.register_module()
class DAVISMultipleGTMixCompressDataset_EDVR(DAVISMultipleGTMixCompressDataset):
 
    def load_annotations(self):
        """Load annotations for REDS dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # generate keys
        # keys = [i.name for i in os.scandir(self.lq_folder)]
        keys = []
        for fin in os.scandir(self.lq_folder):
            img_list=sorted(glob.glob(f'{self.lq_folder}/{fin.name}/*.png'))
            keys.extend(img_list)
 
        data_infos = []
        for key in keys:
            data_infos.append(
                dict(
                    cprs15_path = self.cprs15_folder,
                    cprs25_path = self.cprs25_folder,
                    cprs35_path = self.cprs35_folder,
                    lq_path=self.lq_folder,
                    gt_path=self.gt_folder,
                    key=key,
                    sequence_length=20,  # REDS has 100 frames for each clip
                    num_input_frames=self.num_input_frames))

 
        
        return data_infos
