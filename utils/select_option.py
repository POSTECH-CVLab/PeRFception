from dataloader.litdata import (
    LitDataBlender, LitDataCo3D, LitDataLLFF, LitDataNSVF, LitDataTnT
)
from model.nerf_torch.model import LitNeRF
from model.plenoxel_torch.model import LitPlenoxel
from typing import *

def select_model(
    model_name: str,
):

    if model_name in ["nerf", "jaxnerf"]:
        return LitNeRF()

    elif model_name == "plenoxel":
        return LitPlenoxel()

    else:
        raise f"Unknown model named {model_name}"


def select_dataset(
    dataset_name: str,
    datadir: str, 
    scene_name: str, 
    accelerator: str,
    num_gpus: int,
    num_tpus: int,
):
    if dataset_name == "blender":
        data_fun = LitDataBlender
    elif dataset_name == "llff":
        data_fun = LitDataLLFF
    elif dataset_name == "nsvf":
        data_fun = LitDataNSVF
    elif dataset_name == "tanks_and_temples":
        data_fun = LitDataTnT
    elif dataset_name == "co3d":
        data_fun = LitDataCo3D
    elif dataset_name == "scannet":
        data_fun = None

    return data_fun(
        datadir=datadir, 
        scene_name=scene_name, 
        accelerator=accelerator,
        num_gpus=num_gpus,
        num_tpus=num_tpus,
    )

def select_callback(model_name):

    callbacks = []
    
    if model_name == "plenoxel_torch":
        import model.plenoxel_torch.model as model
        callbacks += [model.ResampleCallBack()]

    return callbacks
