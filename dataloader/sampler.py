import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import torch.distributed as dist


class DDPSequnetialSampler(SequentialSampler): 
    def __init__(self, batch_size, num_replicas, rank, N_total, tpu):
        self.data_source=None
        self.batch_size = batch_size
        self.N_total = N_total
        self.drop_last=False
        ngpus = torch.cuda.device_count()
        if ngpus == 1 and not tpu:
            rank, num_replicas = 0, 1
        else:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        idx_list = np.arange(self.N_total)
        return iter(idx_list[self.rank::self.num_replicas])

    def __len__(self):
        return int(np.ceil(self.N_total / self.num_replicas))


class SingleImageDDPSampler:
    def __init__(self, batch_size, num_replicas, rank, N_img, N_pixels, i_validation, tpu):
        self.batch_size = batch_size
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.drop_last = False
        self.i_validation = i_validation
        ngpus = torch.cuda.device_count()
        if ngpus == 1 and not tpu:
            rank, num_replicas = 0, 1
        else:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Require distributed package to be available")
                rank = dist.get_rank()
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        image_choice = np.random.choice(np.arange(self.N_img),
                                        self.i_validation,
                                        replace=True)
        idx_choice = [
            np.random.choice(np.arange(self.N_pixels), self.batch_size) \
                for _ in range(self.i_validation)
        ]
        for (image_idx, idx) in zip(image_choice, idx_choice):
            idx_ret = image_idx * self.N_pixels + idx
            yield idx_ret[self.rank::self.num_replicas]

    def __len__(self):
        return self.i_validation


class MultipleImageDDPSampler(DistributedSampler):
    def __init__(self, batch_size, num_replicas, rank, total_len, i_validation, tpu):
        self.batch_size = batch_size
        self.total_len = total_len
        self.i_validation = i_validation
        self.drop_last = False        
        ngpus = torch.cuda.device_count()
        if ngpus == 1 and not tpu:
            rank, num_replicas = 0, 1
        else:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError(
                        "Require distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError(
                        "Require distributed package to be available")
                rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        full_index = np.arange(self.total_len)
        indices = [
            np.random.choice(full_index, self.batch_size) \
                for _ in range(self.i_validation)
        ]
        for batch in indices:
            yield batch[self.rank::self.num_replicas]

    def __len__(self):
        return self.i_validation


class RaySet(Dataset):

    def __init__(self, images=None, rays=None):
        self.images = images
        self.images_exist = self.images is not None
        assert rays is not None
        self.rays = rays
        self.N = len(rays)

    def __getitem__(self, index):
        ret = {"ray": torch.from_numpy(self.rays[index])}
        if self.images_exist: 
            ret["target"] = torch.from_numpy(self.images[index])
        return ret

    def __len__(self):
        return self.N
