import argparse
import os
import sys

from utils.SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument("--input_path", required=True, help="path to sens file to read")
parser.add_argument("--output_path", required=True, help="path to output folder")
parser.add_argument(
    "--export_depth_images", dest="export_depth_images", action="store_true"
)
parser.add_argument(
    "--export_color_images", dest="export_color_images", action="store_true"
)
parser.add_argument("--export_poses", dest="export_poses", action="store_true")
parser.add_argument(
    "--export_intrinsics", dest="export_intrinsics", action="store_true"
)
parser.set_defaults(
    export_depth_images=False,
    export_color_images=False,
    export_poses=False,
    export_intrinsics=False,
)

opt = parser.parse_args()
print(opt)


def main(scene_name):
    print(f"processing {scene_name}")
    sens_file = os.path.join(opt.input_path, scene_name, f"{scene_name}.sens")
    outpath = os.path.join(opt.output_path, scene_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # load the data
    imsize = [480, 640]
    sys.stdout.write("loading %s..." % sens_file)
    sd = SensorData(sens_file)
    sys.stdout.write("loaded!\n")
    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(outpath, "depth"), image_size=imsize)
    if opt.export_color_images:
        sd.export_color_images(os.path.join(outpath, "color"), image_size=imsize)
    if opt.export_poses:
        sd.export_poses(os.path.join(outpath, "pose"))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(outpath, "intrinsic"), image_size=imsize)


if __name__ == "__main__":
    from multiprocessing import Pool

    scene_names = os.listdir("/root/data/scannet/scans")
    scene_names = sorted(scene_names)

    pool = Pool(processes=16)
    pool.map(main, scene_names)
    pool.close()
    pool.join()
