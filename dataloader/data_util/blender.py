import numpy as np
import torch
import json
import imageio
import os

from dataloader.spherical_poses import spherical_poses


def load_blender_data(
    datadir: str, 
    scene_name: str,
    test_skip: int, 
    cam_scale_factor: float,
    white_bkgd: bool,
):
    basedir = os.path.join(datadir, scene_name)
    cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    images = []
    extrinsics = []
    counts = [0]        
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or test_skip==0:
            skip = 1
        else:
            skip = test_skip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        images.append(imgs)
        extrinsics.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    images = np.concatenate(images, 0)

    extrinsics = np.concatenate(extrinsics, 0)

    extrinsics[:, :3, 3] *= cam_scale_factor
    extrinsics = extrinsics @ cam_trans

    h, w = imgs[0].shape[:2]
    num_frame = len(extrinsics)
    i_split += [np.arange(num_frame)]

    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * w / np.tan(.5 * camera_angle_x)
    intrinsics = np.array(
        [
            [
                [focal, 0., 0.5 * w],
                [0., focal, 0.5 * h],
                [0., 0., 1.]
            ] for _ in range(num_frame)
        ]
    )
    image_sizes = np.array([[h, w] for _ in range(num_frame)])

    render_poses = spherical_poses(cam_trans)
    render_poses[:, :3, 3] *= cam_scale_factor
    near = 2.
    far = 6.

    if white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    return (
        images, 
        intrinsics, 
        extrinsics,
        image_sizes,
        near,
        far,
        (-1, -1),
        i_split,
        render_poses
    )