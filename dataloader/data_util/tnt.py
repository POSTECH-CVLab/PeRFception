import os
import numpy as np
import glob
import imageio


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


def load_tnt_data(datadir):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    # camera parameters files
    intrinsics_files = find_files('{}/train/intrinsics'.format(datadir), exts=['*.txt'])
    pose_files = find_files('{}/train/pose'.format(datadir), exts=['*.txt'])
    pose_files += find_files('{}/validation/pose'.format(datadir), exts=['*.txt'])
    pose_files += find_files('{}/test/pose'.format(datadir), exts=['*.txt'])
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(datadir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # assume all images have the same size as training image
    train_imgfile = find_files('{}/train/rgb'.format(datadir), exts=['*.png', '*.jpg'])
    val_imgfile = find_files('{}/validation/rgb'.format(datadir), exts=['*.png', '*.jpg'])
    test_imgfile = find_files('{}/test/rgb'.format(datadir), exts=['*.png', '*.jpg'])
    i_train = np.arange(len(train_imgfile))
    i_val = np.arange(len(val_imgfile)) + len(train_imgfile)
    i_test = np.arange(len(test_imgfile)) + len(train_imgfile) + len(val_imgfile)
    i_split = (i_train, i_val, i_test)

    im = np.stack([imageio.imread(imgfile) for imgfile in train_imgfile + val_imgfile + test_imgfile]) / 255.
    H, W = im[0].shape[:2]

    intrinsics = parse_txt(intrinsics_files[0])
    poses = np.stack([parse_txt(pose_file) for pose_file in pose_files])

    return im, poses, poses, (H, W), intrinsics, i_split