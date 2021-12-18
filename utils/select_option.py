from model.jaxnerf_torch.model import LitJaxNeRFLLFF, LitJaxNeRFBlender

from dataloader.llff import LitLLFF
from dataloader.blender import LitBlender


def select_model(args):

    if args.model == "jaxnerf_torch":

        if args.dataset_type == "blender":
            return LitJaxNeRFBlender
        elif args.dataset_type == "llff":
            return LitJaxNeRFLLFF
        else:
            raise f"Unknown dataset named {args.dataset}"
