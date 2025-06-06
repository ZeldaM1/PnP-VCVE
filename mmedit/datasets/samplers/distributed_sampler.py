# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import math
import numpy as np
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from mmedit.core.utils import sync_random_seed


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 samples_per_gpu=1,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        # fix the bug of the official implementation
        self.num_samples_per_replica = int(
            math.ceil(
                len(self.dataset) * 1.0 / self.num_replicas / samples_per_gpu))
        self.num_samples = self.num_samples_per_replica * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

        # to avoid padding bug when meeting too small dataset
        if len(dataset) < self.num_replicas * samples_per_gpu:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'sampler cannot pad your dataset correctly. We highly '
                'recommend you to use fewer GPUs to finish your work')

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedWeightedSampler(DistributedSampler):
    """DistributedSampler inheriting from `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,weights=[], replacement=True, **kwargs):
        super().__init__(**kwargs)
        
        self.weights=np.array(weights)
        self.replacement=replacement
        assert len(self.dataset.datasets)==len(weights)

    def __iter__(self):
        
        weights_used = torch.full((1,self.total_size), 0)
        start_idx=0
        for i in range(len(self.dataset.datasets)):
            length=len(self.dataset.datasets[i])
            end_idx=start_idx+length
            # print('-----------', start_idx,end_idx,self.weights[i])
            weights_used[:,start_idx:end_idx]=torch.full((1,length), self.weights[i])
            start_idx=end_idx
        # breakpoint()
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.multinomial(
                input=torch.as_tensor(weights_used[0], dtype=torch.double),
                num_samples=self.total_size,
                replacement=self.replacement,
                generator=g
            ).tolist()

        else:
            indices = torch.arange(len(self.dataset)).tolist()
 

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # breakpoint()

        return iter(indices)
     
 