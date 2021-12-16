from model.jaxnerf_torch.model import LitJaxNeRF
from model.snerg_torch.model import LitSNeRGNeRF

from dataloader.llff import LitLLFF
from dataloader.blender import LitBlender

def select_model(args):

    if args.model == "jaxnerf_torch":
        return LitJaxNeRF
    elif args.model == "snerg_torch":
        return LitSNeRGNeRF

def select_dataloader(args):

    if args.dataset_type == "llff":
        return LitLLFF(args)
    elif args.dataset_type == "blender":
        return LitBlender(args)