import os, glob
import numpy as np
import imageio
import gzip
import json
import cv2

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
    trans_t = lambda t : np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1]], dtype=np.float32
    )
    rot_phi = lambda phi : np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]], dtype=np.float32
    )
    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]], dtype=np.float32
    )
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array(
        [
            [-1,0,0,0],
            [0,0,1,0],
            [0,1,0,0],
            [0,0,0,1]
        ], dtype=np.float32
    ) @ c2w
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


def load_co3d_data(datadir, cam_scale_factor=0.95):

    datadir = datadir.rstrip("/")

    cam_trans = np.diag(np.array([-1, 1, 1, 1], dtype=np.float32))

    imgdir = os.path.join(datadir, "images")
    imgpath = [os.path.join(imgdir, fpath) for fpath in sorted(os.listdir(imgdir))]
    imgs = [imageio.imread(imgfile) for imgfile in imgpath]
    H, W = imgs[0].shape[:2]
    if H > 800 or W > 800:
        if H > W:
            H, W = 800, int(800 * W / H)
        else:
            H, W = int(800 * H / W), 800
    imgs = np.stack([np.asarray(cv2.resize(img, dsize=(W, H))) for img in imgs]) / 255.

    scene_number = datadir.split("/")[-1]
    path_to_category = "/".join(datadir.split("/")[:-1])

    json_path = os.path.join(path_to_category, "frame_annotations.jgz")
    with gzip.open(json_path, "r") as fp:
        all_frames_data = json.load(fp)

    frame_data = []

    for temporal_data in all_frames_data:
        if temporal_data["sequence_name"] == scene_number:
            frame_data.append(temporal_data)

    assert len(frame_data) == len(imgs)

    frame_data = sorted(frame_data, key=lambda x: x["frame_number"])
    focal = frame_data[0]["viewpoint"]["focal_length"]
    pp = frame_data[0]["viewpoint"]["principal_point"]
    fx, fy = W * focal[0] / 2 , H * focal[1] / 2
    cx = (1 - pp[0]) * W / 2 
    cy = (1 - pp[1]) * H / 2 

    poses = []

    for frame in frame_data:
        pose = np.eye(4)
        R = np.array(frame["viewpoint"]["R"])
        T = np.array(frame["viewpoint"]["T"])
        pose[:3, :3] = R
        pose[:3, 3:] = -R @ T[..., None]
        poses.append(pose)

    poses = np.stack(poses)

    intrinsics = np.array(
        [
            [fx, 0, cx, 0], 
            [0, fy, cy, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], dtype=np.float32
    )

    T, sscale = similarity_from_cameras(poses)

    poses = np.einsum("nij, ki -> nkj", poses, T)
    poses = np.einsum("nij, ki -> nkj", poses, cam_trans)
    scene_scale = cam_scale_factor * sscale
    poses[:, :3, 3] *= scene_scale

    i_split = np.arange(len(imgs))
    i_test = i_split[::10]
    i_train = np.array([i for i in i_split if not i in i_test])

    render_poses = np.stack([
        cam_trans @ pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]
    ], 0)

    return imgs, poses, render_poses, (H, W), intrinsics, (i_train, i_test, i_test)
