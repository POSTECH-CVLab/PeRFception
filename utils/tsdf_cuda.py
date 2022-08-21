import argparse
import os

import numpy as np
import open3d as o3d
import open3d.core as o3c
import tqdm

from dataloader.data_util.common import connected_component_filter, find_files


def integrate(scene_name, outdir, reso=512, max_depth=4.5):
    device = o3c.Device(o3c.Device.CUDA, 0)
    print(f"processing {scene_name}")

    # setup dir
    scenedir = os.path.join(outdir, scene_name)
    if not os.path.exists(scenedir):
        os.makedirs(scenedir, exist_ok=True)

    if os.path.exists(os.path.join(scenedir, f"tsdf_pcd_{reso}.npy")):
        print(f"skip exist {scene_name}")
        return

    files = find_files(os.path.join(scenedir, "color"), exts=["*.jpg"])
    if len(files) == 0:
        print(f"{scenedir} does not contain color images. skip.")
        return
    frame_ids = sorted([os.path.basename(f).rstrip(".jpg") for f in files])
    frame_ids = np.array(frame_ids)

    # filter invalid poses
    poses = np.stack(
        [np.loadtxt(os.path.join(scenedir, "pose", f"{f}.txt")) for f in frame_ids],
        axis=0,
    )
    poses = poses.astype(np.float32)
    numerics = np.all(
        (
            ~np.isinf(poses)
            * ~np.isnan(poses)
            * ~np.isneginf(poses)
            * (np.abs(poses) < 30)
        ).reshape(-1, 16),
        axis=1,
    )
    poses = poses[numerics]
    frame_ids = frame_ids[numerics]

    skip_frame = 1
    if len(frame_ids) > 3000:
        skip_frame = 2
    if len(frame_ids) > 5000:
        skip_frame = 3

    depth_shift = 1000.0

    # load intrinsics
    print(f"loading intrinsic")
    _intrinsic = np.loadtxt(os.path.join(scenedir, "intrinsic", "intrinsic_color.txt"))
    _intrinsic = _intrinsic.astype(np.float32)

    poses = poses[::skip_frame]
    frame_ids = frame_ids[::skip_frame]

    # setup voxel block grid
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1), (1)),
        voxel_size=3.0 / reso,
        block_resolution=16,
        block_count=100000,
        device=device,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        640,
        480,
        _intrinsic[0, 0],
        _intrinsic[1, 1],
        _intrinsic[0, 2],
        _intrinsic[1, 2],
    )
    intrinsic_tensor = o3c.Tensor(intrinsic.intrinsic_matrix, o3c.Dtype.Float64)

    for i, (fid, E) in tqdm.tqdm(
        enumerate(zip(frame_ids, poses)), total=len(frame_ids)
    ):
        # print(f"integraing frame {i+1}/{len(frame_ids)} for scene {scene_name}")
        depth = o3d.t.io.read_image(os.path.join(scenedir, "depth", f"{fid}.png")).to(
            device
        )
        extrinsic = o3c.Tensor(E, o3c.Dtype.Float64)
        extrinsic = o3c.inv(extrinsic).contiguous()
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic_tensor, extrinsic, depth_shift, max_depth
        )

        vbg.integrate(
            frustum_block_coords,
            depth,
            intrinsic_tensor,
            extrinsic,
            depth_shift,
            max_depth,
        )

    # extract geometery
    pcd_tensor = vbg.extract_point_cloud()
    pcd = pcd_tensor.to_legacy()
    xyz = np.asarray(pcd.points)
    sel = connected_component_filter(xyz, 0.05)

    points = np.asarray(pcd.points)[sel].astype(np.float32)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[sel].astype(np.float32)

    np.save(os.path.join(scenedir, f"tsdf_pcd_{reso}.npy"), points)
    print(f">> processed {scene_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--max_depth", type=float, default=4.5)
    parser.add_argument("--reso", type=int, default=1024)
    parser.add_argument("--outdir", type=str, default="/root/data/scannet/scans")
    parser.add_argument("--offset", type=int, default=0)

    args = parser.parse_args()

    if args.scene_name == "all":
        scene_list = sorted(os.listdir(args.outdir))
    else:
        scene_list = [args.scene_name]

    for scene in scene_list:
        integrate(
            scene,
            outdir=args.outdir,
            reso=args.reso,
            max_depth=args.max_depth,
        )
