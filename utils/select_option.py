import os
from typing import *

import gdown

from dataloader.litdata import LitDataCo3D, LitDataScannet
from model.plenoxel_torch.model import LitPlenoxel, ResampleCallBack

url_co3d_list = "https://drive.google.com/uc?id=1jCDaA41ZddkgPl4Yw2h-XI7mt9o56kb7"

def select_model(
    model_name: str,
):  
    return LitPlenoxel()


def select_dataset(
    dataset_name: str,
    datadir: str, 
    scene_name: str, 
    accelerator: str,
    num_gpus: int,
    num_tpus: int,
):
    if dataset_name == "co3d":
        data_fun = LitDataCo3D
        co3d_list_json_path = os.path.join("dataloader/co3d_lists/co3d_list.json")
        if not os.path.exists(co3d_list_json_path):
            gdown.download(url_co3d_list, co3d_list_json_path)
    elif dataset_name == "scannet":
        data_fun = LitDataScannet

    return data_fun(
        datadir=datadir, 
        scene_name=scene_name, 
        accelerator=accelerator,
        num_gpus=num_gpus,
        num_tpus=num_tpus,
    )

def select_callback(model_name):

    return [ResampleCallBack()]
