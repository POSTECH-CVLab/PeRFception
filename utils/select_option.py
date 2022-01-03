from dataloader.llff import LitLLFF
from dataloader.blender import LitBlender


def select_model(model_name, dataset_type):

    if model_name == "jaxnerf_torch":
        import model.jaxnerf_torch.model as model
        if dataset_type == "blender":
            return model.LitJaxNeRFBlender
        elif dataset_type == "llff":
            return model.LitJaxNeRFLLFF
        else:
            raise f"Unknown dataset named {dataset_type}"

    elif model_name == "plenoxel_torch":
        import model.plenoxel_torch.model as model
        if dataset_type == "blender":
            return model.LitPlenoxelBlender
        elif dataset_type == "llff":
            return model.LitPlenoxelLLFF
        else:
            raise f"Unknown dataset named {dataset_type}"

    else:
        raise f"Unknown model named {model_name}"


def select_callback(callbacks, model_name, args):

    if model_name == "plenoxel_torch":
        import model.plenoxel_torch.model as model
        callbacks += [model.ResampleCallBack(args)]

    return callbacks