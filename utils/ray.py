import numpy as np
import torch

def convert_to_ndc(
    origins,
    directions,
    ndc_coeffs,
    near: float = 1.0
):
    """Convert a set of rays to NDC coordinates."""
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz
    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)

    return origins, directions

def batchified_get_rays(h, w, intrinsics, extrinsics, use_pixel_centers): 
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
    rays_o = np.tile(
        extrinsics[:, np.newaxis, :3, 3], (1, h * w, 1)
    ).reshape(-1, 3)
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    tofloat32 = lambda x : x.astype(np.float32)
    rays_o, rays_d = tofloat32(rays_o), tofloat32(rays_d)

    return rays_o, rays_d