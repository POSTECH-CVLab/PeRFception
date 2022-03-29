# Extended NSVF-format dataset loader
# This is a more sane format vs the NeRF formats

import os
import imageio
from tqdm import tqdm
import numpy as np

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
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
        [[0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0]]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1+c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])


    #  R_align = np.eye(3) # DEBUG
    R = (R_align @ R)
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


def load_nsvf_data(
    datadir: str,
    scene_name: str, 
    test_skip: int,
    cam_scale_factor: float,
    white_bkgd: bool,
    data_bbox_scale: float,
):

    basedir = os.path.join(datadir, scene_name)
    img_files = sorted(os.listdir(os.path.join(basedir, "rgb")))
    i_train = [i for i in range(len(img_files)) if img_files[i].startswith("0_")]
    i_val = [i for i in range(len(img_files)) if img_files[i].startswith("1_")]
    i_test = [i for i in range(len(img_files)) if img_files[i].startswith("2_")]

    if len(i_test) == 0:
        i_test = i_val

    if test_skip > 0: 
        i_val = i_val[::test_skip]
        i_test = i_test[::test_skip]

    i_all = i_train + i_val + i_test

    img_files = [img_files[i] for i in i_all]

    poses = []
    image_sizes = []
    images = []

    for img_fname in tqdm(img_files):
        img_path = os.path.join(basedir, "rgb", img_fname)
        image = imageio.imread(img_path)
        full_size = list(image.shape[:2])
        pose_fname = os.path.splitext(img_fname)[0] + ".txt"
        pose_path = os.path.join(basedir, "pose", pose_fname)

        cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
        if len(cam_mtx) == 3:
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
            cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)

        poses.append(cam_mtx)  # C2W (4, 4) OpenCV
        image_sizes.append(full_size)
        images.append(image)

    poses = np.stack(poses)

    # bbox_path = os.path.join(basedir, "bbox.txt")
    # if os.path.exists(bbox_path):
    #     bbox_data = np.loadtxt(bbox_path)
    #     center = (bbox_data[:3] + bbox_data[3:6]) * 0.5
    #     radius = (bbox_data[3:6] - bbox_data[:3]) * 0.5 * data_bbox_scale

    #     # Recenter
    #     poses[:, :3, 3] -= center
    #     # Rescale
    #     scene_scale = 1.0 / radius.max()

    # Select subset of files
    T, sscale = similarity_from_cameras(poses)

    poses = T @ poses
    scene_scale = cam_scale_factor * sscale
    poses[:, :3, 3] *= scene_scale

    images = np.stack(images) / 255.0

    if white_bkgd:
        # Apply alpha channel
        images = images[..., :3] * images[..., 3:] + (1.0 - images[..., 3:])
    else:
        images = images[..., :3]

    assert full_size[0] > 0 and full_size[1] > 0, "Empty images"

    intrin_path = os.path.join(basedir, "intrinsics.txt")
    with open(intrin_path, "r") as f:
        spl = f.readline().split()
        fx = fy = float(spl[0])
        cx = float(spl[1])
        cy = float(spl[2])

    intrinsics = np.stack([
        np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ]) for _ in i_all
    ])
    extrinsics = poses
    image_sizes = np.array(image_sizes)
    near, far = 0, 1
    ndc_coeffs = (-1, -1)

    i_train = np.arange(len(i_train))
    i_val = np.arange(len(i_val)) + len(i_train)
    i_test = np.arange(len(i_test)) + len(i_train) + len(i_val)
    i_all = np.arange(len(i_train) + len(i_val) + len(i_test))

    i_split = (i_train, i_val, i_test, i_all)
    
    render_poses = np.stack(
        [pose_spherical(angle, -30.0, 4.0)
        for angle in np.linspace(-180,180,40+1)[:-1]], 0
    )
    render_poses[:, :3, 3] *= scene_scale

    return (
        images, 
        intrinsics, 
        extrinsics,
        image_sizes,
        near,
        far,
        ndc_coeffs,
        i_split,
        render_poses
    )

