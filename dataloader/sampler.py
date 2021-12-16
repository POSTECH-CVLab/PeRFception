import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class SingleImageSampler:

    def __init__(self, batch_size, N_img, N_pixels, i_validation, tpu_num):
        self.batch_size = batch_size
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.drop_last = False
        self.i_validation = i_validation
        self.tpu_num = tpu_num

    def __iter__(self):
        image_choice = np.random.choice(
            np.arange(self.N_img), self.i_validation, replace=True
        )
        idx_choice = [
            np.random.choice(np.arange(self.N_pixels), self.batch_size) \
                for _ in range(self.i_validation)
        ]
        for (image_idx, idx) in zip(image_choice, idx_choice):
            idx_ret = image_idx * self.N_pixels + idx
            for ray_num in idx_ret:
                yield ray_num

    def __len__(self):
        return self.i_validation * self.batch_size // self.tpu_num


class SingleImageDDPSampler(DistributedSampler):

    def __init__(self, batch_size, N_img, N_pixels, i_validation):
        self.batch_size = batch_size
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.drop_last = False
        self.i_validation = i_validation

    def __iter__(self): 
        image_choice = np.random.choice(
            np.arange(self.N_img), self.i_validation, replace=True
        )
        idx_choice = [
            np.random.choice(np.arange(self.N_pixels), self.batch_size) \
                for _ in range(self.i_validation)
        ]
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        for (image_idx, idx) in zip(image_choice, idx_choice):
            idx_ret = image_idx * self.N_pixels + idx
            yield idx_ret[rank::num_replicas]

    def __len__(self):
        return self.i_validation


class MultipleImageSampler:

    def __init__(self, batch_size, total_len, i_validation, tpu_num):
        self.batch_size = batch_size
        self.total_len = total_len
        self.i_validation = i_validation
        self.tpu_num = tpu_num

    def __iter__(self): 
        full_index = np.arange(self.total_len)
        indices = [
            np.random.choice(full_index, self.batch_size) \
                for _ in range(self.i_validation)
        ]
        for batch in indices:
            for idx in batch:
                yield idx

    def __len__(self):
        return self.i_validation * self.batch_size // self.tpu_num

class MultipleImageDDPSampler:

    def __init__(self, batch_size, total_len, i_validation):
        self.batch_size = batch_size
        self.total_len = total_len
        self.i_validation = i_validation

    def __iter__(self): 
        full_index = np.arange(self.total_len)
        indices = [
            np.random.choice(full_index, self.batch_size) \
                for _ in range(self.i_validation)
        ]
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()   
        for batch in indices:
            yield batch[rank::num_replicas]

    def __len__(self):
        return self.i_validation 


class RaySet(Dataset):

    def __init__(self, images, rays):
        self.images = images
        self.rays = rays
        self.N = len(images)

    def __getitem__(self, index):
        return {
            "target": torch.from_numpy(self.images[index]), 
            "ray": torch.from_numpy(self.rays[index])
        }

    def __len__(self):
        return len(self.images)