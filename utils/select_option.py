from jaxnerf_torch.model import LitJaxNeRF
from dataloader.llff import LitLLFF 

def select_model(args):

    if args.model == "jaxnerf_torch":
        return LitJaxNeRF

def select_dataloader(args):

    if args.dataset_type == "llff":
        return LitLLFF(args)