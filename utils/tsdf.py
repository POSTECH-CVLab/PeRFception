import argparse
import os

import numpy as np
import open3d as o3d
import tqdm

from dataloader.data_util.scannet import SensorData


def integrate(
    outdir,
    scene_name,
    max_frame,
    skip_frame,
    blur_thresh,
    max_image_dim,
    voxel_size,
    max_depth=4.5,
    debug=False,
):
    # setup exporter
    filepath = f"./data/scannet/{scene_name}/{scene_name}.sens"
    exporter = SensorData(
        filepath, max_frame=max_frame, preview=False, blur_thresh=blur_thresh
    )

    # resize images
    H, W = exporter.color_height, exporter.color_width
    max_hw = max(H, W)
    resize_scale = max_image_dim / max_hw
    imsize = [int(round(resize_scale * H, -1)), int(round(resize_scale * W, -1))]

    # filter invalid poses
    poses = exporter.export_poses(skip_frame)
    numerics = np.all(
        (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
        axis=1,
    )
    images, masks, _ = exporter.export_color_images(
        frame_skip=skip_frame, image_size=imsize, pose_mask=numerics
    )
    depths = exporter.export_depth_images(frame_skip=skip_frame, image_size=imsize)
    images = images.astype(np.float32) / 255.0
    images, poses, depths = images[masks], poses[masks], depths[masks]

    # setup TSDF volume
    _intrinsic, _ = exporter.export_intrinsics()
    _intrinsic *= resize_scale
    _intrinsic[[2, 3], [2, 3]] = 1
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        imsize[1],
        imsize[0],
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
    for image, pose, depth in tqdm.tqdm(zip(images, poses, depths)):
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
    mesh = volume.extract_triangle_mesh()
    pcd = volume.extract_point_cloud()

    o3d.io.write_triangle_mesh(os.path.join(outdir, "tsdf_mesh.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(outdir, "tsdf_pcd.pcd"), pcd)
    if debug:
        voxel = volume.extract_voxel_point_cloud()
        o3d.io.write_point_cloud(os.path.join(outdir, "voxel.pcd"), voxel)


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.scene_name == "all":
        scene_list = sorted(os.listdir(args.outdir))
    else:
        scene_list = [args.scene_name]

    for scene in scene_list:
        outdir = os.path.join(args.outdir, scene)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        integrate(
            outdir,
            args.scene_name,
            args.max_frame,
            args.skip_frame,
            args.blur_thresh,
            args.max_image_dim,
            args.voxel_size,
            args.max_depth,
        )
