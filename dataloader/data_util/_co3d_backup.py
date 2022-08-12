import glob
import gzip
import json
import os

import cv2
import numpy as np
import scipy as sp
import gin
import torch
import torch_cluster as tc

from dataloader.random_pose import random_pose, pose_interp
from dataloader.spherical_poses import spherical_poses


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


def similarity_from_cameras(c2w, fix_rot=False):
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

    if fix_rot:
        R_align = np.eye(3)
        R = np.eye(3)
    else:
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
    render_scene_interp: bool = False,
    render_random_pose: bool = True,
    interp_fac: int = 5,
    v2_mode: bool = False
):

    with open("dataloader/co3d_lists/co3d_list.json") as fp:
        co3d_lists = json.load(fp)

    datadir = datadir.rstrip("/")
    cls_name = co3d_lists[scene_name]
    basedir = os.path.join(datadir, cls_name, scene_name)
    cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
    # cam_trans = np.diag(np.array([1, 1, 1, 1], dtype=np.float32))

    scene_number = basedir.split("/")[-1]

    json_path = os.path.join(basedir, "..", "frame_annotations.jgz")
    with gzip.open(json_path, "r") as fp:
        all_frames_data = json.load(fp)
    
    # json_path_v1 = os.path.join(basedir, "..", "frame_annotations.jgz")
    # json_path_v2 = os.path.join("data", "co3d_v2", cls_name,"frame_annotations.jgz")
    # with gzip.open(json_path_v1, "r") as fp:
    #     all_frames_data1 = json.load(fp)

    # with gzip.open(json_path_v2, "r") as fp:
    #     all_frames_data2 = json.load(fp)

    frame_data, images, intrinsics, extrinsics, image_sizes = [], [], [], [], []
    # frame_data1, frame_data2, images, intrinsics, extrinsics, image_sizes = [], [], [], [], [], [], []

    for temporal_data in all_frames_data:
        if temporal_data["sequence_name"] == scene_number:
            frame_data.append(temporal_data)

    # for temporal_data in all_frames_data1:
    #     if temporal_data["sequence_name"] == scene_number:
    #         frame_data1.append(temporal_data)

    # for temporal_data in all_frames_data2:
    #     if temporal_data["sequence_name"] == scene_number:
    #         frame_data2.append(temporal_data)

    used = []
    for (i, frame) in enumerate(frame_data):
        img = cv2.imread(os.path.join(datadir, frame["image"]["path"]))
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
        scale_arr = np.array([W * 0.5, H * 0.5], dtype=np.float32) if not v2_mode else \
            np.array([W * 0.5, W * 0.5], dtype=np.float32)
        focal = fxy * scale_arr * scale
        prp = (
            -1.0 * (cxy - 1.0) * scale_arr * scale
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

        if any([np.all(pose == _pose) for _pose in extrinsics]):
            continue

        used.append(i)
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

    if inlier.sum() != 0: 
        intrinsics = intrinsics[inlier]
        extrinsics = extrinsics[inlier]
        image_sizes = image_sizes[inlier]
        images = [images[i] for i in range(len(inlier)) if inlier[i]]

    extrinsics = np.stack(extrinsics)
    T, sscale = similarity_from_cameras(extrinsics)
    extrinsics = T @ extrinsics
    
    extrinsics[:, :3, 3] *= sscale * cam_scale_factor

    num_frames = len(extrinsics)
    # if v2_mode:
    #     i_all = tc.fps(torch.from_numpy(extrinsics[:, :3, 3]), ratio=0.5)
    #     i_all = np.sort(i_all)
    # else:
    i_all = np.arange(num_frames)
    i_test = i_all[::10]
    i_val = i_test
    i_train = np.array([i for i in i_all if not i in i_test])
    i_split = (i_train, i_val, i_test, i_all)

    if render_random_pose:
        render_poses = random_pose(extrinsics[i_all], 50)
    elif render_scene_interp:
        render_poses = pose_interp(extrinsics[i_all], interp_fac)
    # render_poses = spherical_poses(sscale * cam_scale_factor * np.eye(4))
    
    near, far = 0., 1.
    ndc_coeffs = (-1., -1.)

    label_info = {}
    label_info["T"] = T
    label_info["sscale"] = sscale * cam_scale_factor
    label_info["class_label"] = basedir.rstrip("/").split("/")[-2]
    label_info["extrinsics"] = extrinsics
    label_info["intrinsics"] = intrinsics
    label_info["image_sizes"] = image_sizes

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
        label_info
    )
