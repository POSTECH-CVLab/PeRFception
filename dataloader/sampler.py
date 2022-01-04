import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import torch.distributed as dist

class DDPSampler(SequentialSampler):
    
    def __init__(self, batch_size, num_replicas, rank, tpu):
        self.data_source=None
        self.batch_size = batch_size
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

class DDPSequnetialSampler(DDPSampler): 

    def __init__(self, batch_size, num_replicas, rank, N_total, tpu):
        self.N_total = N_total
        super(DDPSequnetialSampler, self).__init__(batch_size, num_replicas, rank, tpu)

    def __iter__(self):
        idx_list = np.arange(self.N_total)
        return iter(idx_list[self.rank::self.num_replicas])

    def __len__(self):
        return int(np.ceil(self.N_total / self.num_replicas))


class SingleImageDDPSampler(DDPSampler):
    
    def __init__(self, batch_size, num_replicas, rank, N_img, N_pixels, i_validation, tpu):
        super(SingleImageDDPSampler, self).__init__(batch_size, num_replicas, rank, tpu)
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.i_validation = i_validation

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


class MultipleImageDDPSampler(DDPSampler):
    def __init__(self, batch_size, num_replicas, rank, total_len, i_validation, tpu):
        super(MultipleImageDDPSampler, self).__init__(batch_size, num_replicas, rank, tpu)
        self.total_len = total_len
        self.i_validation = i_validation

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


class MultipleImageWOReplaceDDPSampler(MultipleImageDDPSampler):

    def __init__(self, batch_size, num_replicas, rank, total_len, i_validation, tpu):
        super(MultipleImageWOReplaceDDPSampler, self).__init__(
            batch_size, num_replicas, rank, total_len, i_validation, tpu
        )

    def __iter__(self):
        indices = [
            np.random.permutation(self.total_len) \
                for _ in range(int(
                    np.ceil(self.i_validation * self.batch_size / self.total_len)
                ))
        ]
        indices = np.concatenate(indices)[:self.i_validation * self.batch_size]
        indices = indices.reshape(self.i_validation, self.batch_size)

        for batch in indices:
            yield batch[self.rank::self.num_replicas]

    def __len__(self):
        return self.i_validation


class RaySet(Dataset):

    def __init__(self, images=None, rays=None):
        self.images = images
        self.images_exist = self.images is not None
        assert rays is not None
        rays[:, 1] = rays[:, 1] / np.linalg.norm(rays[:, 1], axis=1)[:, np.newaxis]
        self.rays = rays
        
        self.N = len(rays)

    def __getitem__(self, index):
        ret = {"ray": torch.from_numpy(self.rays[index])}
        if self.images_exist: 
            ret["target"] = torch.from_numpy(self.images[index])
        return ret

    def __len__(self):
        return self.N
