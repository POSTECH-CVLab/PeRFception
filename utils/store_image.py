from PIL import Image
import os
import numpy as np


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs, depths):
    for (i, (rgb, depth)) in enumerate(zip(rgbs, depths)):
        imgname = f"image{str(i).zfill(3)}.jpg"
        depthname = f"depth{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb))
        imgpath = os.path.join(dirpath, imgname)
        depthimg = Image.fromarray(norm8b(depth))
        depthpath = os.path.join(dirpath, depthname)
        rgbimg.save(imgpath)
        depthimg.save(depthpath)
