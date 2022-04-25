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
    
    def __init__(
        self, 
        batch_size, 
        num_replicas, 
        rank, 
        N_img, 
        N_pixels, 
        epoch_size, 
        tpu,
        precrop,
        precrop_steps,
    ):
        super(SingleImageDDPSampler, self).__init__(batch_size, num_replicas, rank, tpu)
        self.N_pixels = N_pixels
        self.N_img = N_img
        self.epoch_size = epoch_size
        self.precrop = precrop
        self.precrop_steps = precrop_steps

    def __iter__(self):
        image_choice = np.random.choice(
            np.arange(self.N_img),
            self.epoch_size,
            replace=True
        )
        image_shape = self.N_pixels[image_choice]
        if not self.precrop:
            idx_choice = [
                np.random.choice(np.arange(image_shape[i, 0] * image_shape[i, 1]), self.batch_size)
                for i in range(self.epoch_size)
            ]
        else:
            idx_choice = []
            h_pick = [
                np.random.choice(
                    np.arange(image_shape[i, 0] // 2), self.batch_size
                ) + image_shape[i, 0] // 4 for i in range(self.precrop_steps)
            ]
            w_pick = [
                np.random.choice(
                    np.arange(image_shape[i, 1] // 2), self.batch_size
                ) + image_shape[i, 1] // 4 for i in range(self.precrop_steps)
            ]
            idx_choice = [h_pick[i] * image_shape[i, 1] + w_pick[i] for i in range(self.precrop_steps)]
                
            idx_choice += [
                np.random.choice(np.arange(image_shape[i, 0] * image_shape[i, 1]), self.batch_size) 
                for i in range(self.epoch_size - self.precrop_steps)
            ]
            self.precrop = False

        for ((h, w), image_idx, idx) in zip(image_shape, image_choice, idx_choice):
            idx_ret = image_idx * h * w + idx
            yield idx_ret[self.rank::self.num_replicas]

    def __len__(self):
        return self.epoch_size


class MultipleImageDDPSampler(DDPSampler):
    def __init__(self, batch_size, num_replicas, rank, total_len, epoch_size, tpu):
        super(MultipleImageDDPSampler, self).__init__(batch_size, num_replicas, rank, tpu)
        self.total_len = total_len
        self.epoch_size = epoch_size

    def __iter__(self):
        full_index = np.arange(self.total_len)
        indices = [
            np.random.choice(full_index, self.batch_size) \
                for _ in range(self.epoch_size)
        ]
        for batch in indices:
            yield batch[self.rank::self.num_replicas]

    def __len__(self):
        return self.epoch_size


class MultipleImageWOReplaceDDPSampler(MultipleImageDDPSampler):

    def __init__(self, batch_size, num_replicas, rank, total_len, epoch_size, tpu):
        super(MultipleImageWOReplaceDDPSampler, self).__init__(
            batch_size, num_replicas, rank, total_len, epoch_size, tpu
        )

    def __iter__(self):
        indices = [
            np.random.permutation(self.total_len) \
                for _ in range(int(
                    np.ceil(self.epoch_size * self.batch_size / self.total_len)
                ))
        ]
        indices = np.concatenate(indices)[:self.epoch_size * self.batch_size]
        indices = indices.reshape(self.epoch_size, self.batch_size)

        for batch in indices:
            yield batch[self.rank::self.num_replicas]

    def __len__(self):
        return self.epoch_size


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
