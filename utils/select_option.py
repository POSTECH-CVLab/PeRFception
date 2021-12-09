from jaxnerf_torch.model import LitJaxNeRF
from dataloader.llff import LitLLFF 

def select_trainer(args, info):

    if args.model == "jaxnerf_torch":
        return LitJaxNeRF(args, info)

def select_dataloader(args):

    if args.dataset_type == "llff":
        return LitLLFF(args)