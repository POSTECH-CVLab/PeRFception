import glob
import gzip
import json
import os

import cv2
import numpy as np
import gin

from dataloader.random_pose import random_pose


def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def pose_spherical(theta, phi, radius):
    trans_t = lambda t: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=np.float32
    )
    rot_phi = lambda phi: np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    rot_theta = lambda th: np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        @ c2w
    )
    return c2w


def similarity_from_cameras(c2w):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale


@gin.configurable()
def load_co3d_data(
    datadir: str,
    scene_name: str, 
    max_image_dim: int,
    cam_scale_factor: float,
):

    datadir = datadir.rstrip("/")
    basedir = os.path.join(datadir, scene_name)
    cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))

    scene_number = basedir.split("/")[-1]

    json_path = os.path.join(basedir, "..", "frame_annotations.jgz")
    with gzip.open(json_path, "r") as fp:
        all_frames_data = json.load(fp)

    frame_data, images, intrinsics, extrinsics, image_sizes = [], [], [], [], []

    for temporal_data in all_frames_data:
        if temporal_data["sequence_name"] == scene_number:
            frame_data.append(temporal_data)

    for frame in frame_data:
        img = cv2.imread(os.path.join("data/co3d", frame["image"]["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        H, W = frame["image"]["size"]
        max_hw = max(H, W)
        approx_scale = max_image_dim / max_hw

        if approx_scale < 1.0:
            H2 = int(approx_scale * H)
            W2 = int(approx_scale * W)
            img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
        else:
            H2 = H
            W2 = W

        image_size = np.array([H2, W2])
        scale = np.array([W2 / W, H2 / H], dtype=np.float32)
        fxy = np.array(frame["viewpoint"]["focal_length"])
        cxy = np.array(frame["viewpoint"]["principal_point"])
        focal = fxy * np.array([W * 0.5, H * 0.5], dtype=np.float32) * scale
        prp = (
            -1.0 * (cxy - 1.0) * np.array([W * 0.5, H * 0.5], dtype=np.float32) * scale
        )
        R = np.array(frame["viewpoint"]["R"])
        T = np.array(frame["viewpoint"]["T"])
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3:] = -R @ T[..., None]
        pose = pose @ cam_trans
        intrinsic = np.array(
            [
                [focal[0], 0.0, prp[0], 0.0],
                [0.0, focal[1], prp[1], 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        image_sizes.append(image_size)
        intrinsics.append(intrinsic)
        extrinsics.append(pose)
        images.append(img)

    intrinsics = np.stack(intrinsics)
    extrinsics = np.stack(extrinsics)
    image_sizes = np.stack(image_sizes)

    H_median, W_median = np.median(
        np.stack([image_size for image_size in image_sizes]), axis=0
    )

    H_inlier = np.abs(image_sizes[:, 0] - H_median) / H_median < 0.1
    W_inlier = np.abs(image_sizes[:, 1] - W_median) / W_median < 0.1
    inlier = np.logical_and(H_inlier, W_inlier)
    dists = np.linalg.norm(
        extrinsics[:, :3, 3] - np.median(extrinsics[:, :3, 3], axis=0), axis=-1
    )
    med = np.median(dists)
    good_mask = dists < (med * 5.0)
    inlier = np.logical_and(inlier, good_mask)

    intrinsics = intrinsics[inlier]
    extrinsics = extrinsics[inlier]
    image_sizes = image_sizes[inlier]
    images = [images[i] for i in range(len(inlier)) if inlier[i]]

    extrinsics = np.stack(extrinsics)
    T, sscale = similarity_from_cameras(extrinsics)
    extrinsics = T @ extrinsics
    extrinsics[:, :3, 3] *= sscale * cam_scale_factor

    i_all = np.arange(len(images))
    i_test = i_all[::10]
    i_val = i_train
    i_train = np.array([i for i in i_all if not i in i_test])
    i_split = (i_train, i_val, i_test, i_all)

    render_poses = random_pose(extrinsics)
    
    near, far = 0., 1.
    ndc_coeffs = (-1., -1.)

    return (
        images, 
        intrinsics, 
        extrinsics,
        image_sizes,
        near,
        far,
        ndc_coeffs, 
        i_split,
        render_poses,
    )
