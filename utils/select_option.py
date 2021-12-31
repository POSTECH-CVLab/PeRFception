from model.jaxnerf_torch.model import LitJaxNeRFLLFF, LitJaxNeRFBlender

from dataloader.llff import LitLLFF
from dataloader.blender import LitBlender


def select_model(model_name, dataset_type):

    if model_name == "jaxnerf_torch":

        if dataset_type == "blender":
            return LitJaxNeRFBlender
        elif dataset_type == "llff":
            return LitJaxNeRFLLFF
        else:
            raise f"Unknown dataset named {dataset_type}"
