import glob
import os
import struct
import zlib

import cv2
import imageio
import numpy as np
import plyfile
from tqdm import tqdm

from dataloader.data_util.common import find_files, similarity_from_cameras


def detect_blur_fft(image, size=60, thresh=10):
    if image.ndim > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    cx, cy = (int(w / 2), int(h / 2))
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    fft_shift[cy - size : cy + size, cx - size : cx + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    mag = 20 * np.log(np.abs(recon))
    mean = np.mean(mag)
    return mean, mean < thresh


## Scannet .sens data exporter
## https://github.com/ScanNet/ScanNet/tree/master/SensReader/python
COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename, max_frame, preview=False, blur_thresh=None):
        self.version = 4
        self.filename = filename
        self.max_frame = max_frame
        self.preview = preview
        self.blur_thresh = blur_thresh
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]

            frames_in_use = (
                np.array(
                    [
                        np.floor(num_frames * (i / self.max_frame))
                        for i in range(self.max_frame)
                    ],
                    dtype=np.int,
                )
                if self.max_frame != -1
                else np.arange(num_frames)
            )
            frames_in_use = np.unique(frames_in_use)

            self.num_frames = num_frames
            self.frames = []

            if not self.preview:
                for i in tqdm(range(num_frames), desc="loading frames"):
                    frame = RGBDFrame()
                    frame.load(f)
                    if i in frames_in_use:
                        self.frames.append(frame)

    def export_color_images(self, image_size=None, frame_skip=1, pose_mask=None):
        colors = []
        scores = []
        masks = []
        for f in tqdm(
            range(0, len(self.frames), frame_skip), desc="exporting color images"
        ):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if self.blur_thresh is not None and self.blur_thresh > 0:
                blur, is_blurry = detect_blur_fft(color, thresh=self.blur_thresh)
            else:
                blur, is_blurry = 0, False
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            masks.append(~is_blurry)
            scores.append(blur)
            colors.append(color)

        colors = np.stack(colors, axis=0)
        scores = np.stack(scores, axis=0)
        masks = np.stack(masks, axis=0)

        masks = np.logical_and(masks, pose_mask)

        return colors, masks, scores

    def export_poses(self, frame_skip=1):
        poses = []
        for f in tqdm(range(0, len(self.frames), frame_skip), desc="exporting poses"):
            poses.append(self.frames[f].camera_to_world)
        poses = np.stack(poses, axis=0)
        return poses

    def export_depth_images(self, image_size=None, frame_skip=1):
        depths = []
        for f in tqdm(
            range(0, len(self.frames), frame_skip), desc="exporting depth images"
        ):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            depth = depth.astype(np.float32) / self.depth_shift
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            depths.append(depth)
        depths = np.stack(depths, axis=0)
        return depths

    def export_intrinsics(self):
        intrinsic_color = self.intrinsic_color
        intrinsic_depth = self.intrinsic_depth
        return (intrinsic_color, intrinsic_depth)


def load_scannet_data(
    datadir,
    cam_scale_factor=1.0,
    frame_skip=1,
    max_frame=1000,
    preview=False,
    max_image_dim=800,
    blur_thresh=None,
    use_depth=False,
    use_scans=False,
):
    depths = None

    files = find_files(datadir, exts=["*.sens"])
    assert len(files) > 0, f"{datadir} does not contain .sens data."
    filepath = files[0]
    exporter = SensorData(
        filepath, max_frame=max_frame, preview=preview, blur_thresh=blur_thresh
    )
    if preview:
        return exporter

    # export images and poses
    H, W = exporter.color_height, exporter.color_width
    max_hw = max(H, W)
    resize_scale = max_image_dim / max_hw
    imsize = [int(round(resize_scale * H, -1)), int(round(resize_scale * W, -1))]

    poses = exporter.export_poses(frame_skip=frame_skip)
    numerics = np.all(
        (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
        axis=1,
    )

    imgs, masks, scores = exporter.export_color_images(
        frame_skip=frame_skip, image_size=imsize, pose_mask=numerics
    )
    imgs = imgs.astype(np.float32) / 255.0

    if use_depth:

        depths_scan = exporter.export_depth_images(
            image_size=imsize,
            frame_skip=frame_skip,
        )
        if use_scans:
            depths = depths_scan

    # export intrinsics
    intrinsics, _ = exporter.export_intrinsics()
    intrinsics *= resize_scale
    intrinsics[[2, 3], [2, 3]] = 1

    imgs = imgs[masks]
    poses = poses[masks]
    if use_depth:
        depths = depths[masks]

    scene_name = datadir.rstrip("/").split("/")[-1]
    plyname = scene_name + "_vh_clean_2.ply"
    ply_file = os.path.join(datadir, plyname)

    ply = plyfile.PlyData.read(ply_file)["vertex"]
    pcd = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1)
    pcd_mean = pcd.mean(axis=0, keepdims=True)
    pcd -= pcd_mean
    sscale = 1.0 / np.linalg.norm(pcd, axis=1).max()
    scene_scale = cam_scale_factor * sscale

    poses[:, :3, 3] -= pcd_mean
    T, _ = similarity_from_cameras(poses)
    poses = T @ poses

    pcd = (T[:3, :3] @ pcd.T).T

    pcd_depth = None
    if use_depth:
        pts = []
        pts_per_frame = 5000

        def depth_to_pcd(h, w, z, intrinsic, extrinsic):
            N_img = 1
            i, j = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
                indexing="xy",
            )
            i, j = np.tile(i, (N_img, 1, 1)), np.tile(j, (N_img, 1, 1))
            dirs = np.stack(
                [
                    (i - intrinsic[0][2]) / intrinsic[0][0],
                    (j - intrinsic[1][2]) / intrinsic[1][1],
                    np.ones_like(i),
                ],
                -1,
            )
            sel = z.reshape(-1) > 0
            pts = dirs.reshape(-1, 3)[sel] * z.reshape(-1, 1)[sel]
            pts = (
                np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)), -1)
                @ extrinsic.T
            )
            sel = np.linspace(0, pts.shape[0] - 1, pts_per_frame, dtype=np.int32)
            return pts[sel][:, :3]

        for d, E in zip(depths, poses):
            _depth_pcd = depth_to_pcd(imsize[0], imsize[1], d, intrinsics, E)
            pts.append(_depth_pcd)
        pcd_depth = np.concatenate(pts, axis=0)

        ##### Connected Component Anaylsis
        import cc3d
        import MinkowskiEngine as ME
        import torch

        def connected_component_filter(xyz, voxel_size):
            svoxel, idx, idx_inverse = ME.utils.sparse_quantize(
                xyz / voxel_size, return_index=True, return_inverse=True
            )
            svoxel -= svoxel.min(0, keepdim=True).values
            svoxel = svoxel.long()
            dvoxel = torch.zeros((svoxel + 1).max(0).values.tolist())
            dvoxel[svoxel[:, 0], svoxel[:, 1], svoxel[:, 2]] = 1
            labels_out = cc3d.connected_components(dvoxel.numpy(), connectivity=26)
            labels_out = labels_out[svoxel[:, 0], svoxel[:, 1], svoxel[:, 2]]
            counts = np.bincount(labels_out)
            argmax = np.argmax(counts)
            labels_inverse = labels_out[idx_inverse]
            sel = labels_inverse == argmax
            print(
                f">>>> connected component filtering, from {xyz.shape[0]} to {sel.sum()} <<<<"
            )
            return xyz[sel]

        pcd_depth = connected_component_filter(pcd_depth, 0.05)
        #####
        del pts
        print(f"pcd size: {pcd_depth.shape[0]}, pcd mean: {pcd_depth.mean(0)}")

    poses[:, :3, 3] *= scene_scale

    # filter blurry images
    print(f">> sampled {(masks).sum()} non-blurred images from {masks.shape[0]} images")

    H, W = imgs[0].shape[:2]
    i_split = np.arange(len(imgs))
    i_test = np.unique(np.array([int(i * (len(imgs) / 20)) for i in range(20)]))
    i_train = np.array([i for i in i_split if not i in i_test])

    render_poses = poses[::10]
    render_poses = T @ render_poses
    render_poses[:, :3, 3] *= scene_scale

    store_dict = {
        "poses": poses,
        "T": T,
        "scene_scale": scene_scale,
        "pcd_mean": pcd_mean,
        "pcd": pcd_depth if pcd_depth is not None else pcd,
        "pcd_orig": pcd,
        "class_info": None,
    }

    return (
        imgs,
        poses,
        render_poses,
        (H, W),
        intrinsics,
        (i_train, i_test, i_test),
        depths,
        store_dict,
    )
