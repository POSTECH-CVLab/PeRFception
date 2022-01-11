import numpy as np

def batchfied_get_rays(h, w, intrinsics, extrinsics, use_pixel_centers): 
    center = 0.5 if use_pixel_centers else 0. 
    N_img = extrinsics.shape[0]
    i, j = np.meshgrid(
        np.arange(w, dtype=np.float32) + center, 
        np.arange(h, dtype=np.float32) + center, 
        indexing="xy"
    )
    i, j = np.tile(i, (N_img, 1, 1)), np.tile(j, (N_img, 1, 1))
    dirs = np.stack([
        (i - intrinsics[0][2]) / intrinsics[0][0],
        (j - intrinsics[1][2]) / intrinsics[1][1],
        np.ones_like(i)
    ], -1)
    rays_d = np.einsum(
        "nhwc, nrc -> nhwr", dirs, extrinsics[:, :3, :3]
    ).reshape(-1, 3)
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.tile(
        extrinsics[:, np.newaxis, :3, -1], (1, h * w, 1)
    ).reshape(-1, 3)

    return rays_o, rays_d