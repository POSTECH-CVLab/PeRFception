import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import tqdm

from dataloader.data_util.common import connected_component_filter, find_files
from dataloader.data_util.scannet import detect_blur_fft


def integrate(
    scene_name,
    outdir,
    max_frame,
    skip_frame,
    blur_thresh,
    max_image_dim,
    voxel_size,
    max_depth=4.5,
    debug=False,
):
    print(f"processing {scene_name}")
    # setup dir
    scenedir = os.path.join(outdir, scene_name)
    if not os.path.exists(scenedir):
        os.makedirs(scenedir, exist_ok=True)

    if os.path.exists(os.path.join(scenedir, f"tsdf_pcd_{voxel_size}.pcd")):
        print(f"skip exist {scene_name}")
        return

    files = find_files(os.path.join(scenedir, "color"), exts=["*.jpg"])
    assert len(files) > 0, f"{scenedir} does not contain color images."
    frame_ids = sorted([os.path.basename(f).rstrip(".jpg") for f in files])
    frame_ids = np.array(frame_ids)

    # filter invalid poses
    poses = np.stack(
        [np.loadtxt(os.path.join(scenedir, "pose", f"{f}.txt")) for f in frame_ids],
        axis=0,
    )
    poses = poses.astype(np.float32)
    numerics = np.all(
        (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
        axis=1,
    )

    # load images
    print(f"loading images - {len(frame_ids)}")
    colors = np.stack(
        [cv2.imread(os.path.join(scenedir, "color", f"{f}.jpg")) for f in frame_ids],
        axis=0,
    )
    colors = colors.astype(np.float32) / 255.0

    # load depths
    print(f"loading depths - {len(frame_ids)}")
    depth_shift = 1000.0
    depths = np.stack(
        [
            cv2.imread(
                os.path.join(scenedir, "depth", f"{f}.png"), cv2.IMREAD_UNCHANGED
            )
            for f in frame_ids
        ],
        axis=0,
    )
    depths = depths.astype(np.float32) / depth_shift

    # load intrinsics
    print(f"loading intrinsic")
    _intrinsic = np.loadtxt(os.path.join(scenedir, "intrinsic", "intrinsic_color.txt"))
    _intrinsic = _intrinsic.astype(np.float32)

    # filter blurry images
    print(f"filter blurry images")
    if not os.path.exists(os.path.join(scenedir, "blur.npy")):
        blurness = np.stack(
            [detect_blur_fft(c, thresh=blur_thresh)[0] for c in colors], axis=0
        ).reshape(-1)
        np.save(os.path.join(scenedir, "blur.npy"), blurness)
    else:
        blurness = np.load(os.path.join(scenedir, "blur.npy"))
    num_valid = min(150, int(0.2 * len(frame_ids)))
    ths = np.sort(blurness)[num_valid]
    is_valid = np.logical_and(blurness > ths, numerics)
    print(f"filtered {is_valid.sum()} out of {len(is_valid)} images")

    colors, depths, poses = (
        colors[is_valid][::skip_frame],
        depths[is_valid][::skip_frame],
        poses[is_valid][::skip_frame],
    )
    frame_ids = frame_ids[is_valid][::skip_frame]

    # setup TSDF volume
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        colors.shape[2],
        colors.shape[1],
        _intrinsic[0, 0],
        _intrinsic[1, 1],
        _intrinsic[0, 2],
        _intrinsic[1, 2],
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # integration
    for image, pose, depth in tqdm.tqdm(zip(colors, poses, depths)):
        image *= 255.0
        image = image.astype(np.uint8)
        image_o3d = o3d.geometry.Image(image)
        depth_o3d = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            image_o3d,
            depth_o3d,
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

    # extract geometery
    pcd = volume.extract_point_cloud()
    xyz = np.asarray(pcd.points)
    sel = connected_component_filter(xyz, 0.05)

    points = np.asarray(pcd.points)[sel].astype(np.float32)
    colors = np.asarray(pcd.colors)[sel].astype(np.float32)

    np.savez(
        os.path.join(scenedir, f"tsdf_pcd_{voxel_size}.npz"),
        xyz=points,
        color=colors,
    )
    print(f">> processed {scene_name}")


if __name__ == "__main__":
    from functools import partial
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--max_frame", type=int, default=1000)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--blur_thresh", type=float, default=10)
    parser.add_argument("--max_depth", type=float, default=4.5)
    parser.add_argument("--max_image_dim", type=int, default=640)
    parser.add_argument("--voxel_size", type=float, default=0.025)
    parser.add_argument("--basedir", type=str, default="./tsdf_results")
    parser.add_argument("--outdir", type=str, default="/root/data/scannet/scans")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()

    if args.scene_name == "all":
        scene_list = sorted(os.listdir(args.outdir))
    else:
        scene_list = [args.scene_name]

    if args.scene_name == "all":
        integrate_partial = partial(
            integrate,
            outdir=args.outdir,
            max_frame=args.max_frame,
            skip_frame=args.skip_frame,
            blur_thresh=args.blur_thresh,
            max_image_dim=args.max_image_dim,
            voxel_size=args.voxel_size,
            max_depth=args.max_depth,
        )
        scene_list_cur = scene_list[args.offset :: args.num_workers]
        for scene in scene_list_cur:
            integrate_partial(scene)
    else:
        for scene in scene_list:
            integrate(
                scene,
                args.outdir,
                args.max_frame,
                args.skip_frame,
                args.blur_thresh,
                args.max_image_dim,
                args.voxel_size,
                args.max_depth,
            )
