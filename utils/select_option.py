from dataloader.litdata import (
    LitDataBlender, LitDataLLFF, LitDataTnT
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
        return LitDataBlender(
            datadir=datadir, 
            scene_name=scene_name, 
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )
    if dataset_name == "llff":
        return LitDataLLFF(
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
