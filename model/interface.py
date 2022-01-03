import pytorch_lightning as pl
import os

import torch

class LitModel(pl.LightningModule):

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def test_dataloader(self):
        return self.dataset.test_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def predict_dataloader(self):
        return self.dataset.predict_dataloader()

    def __init__(self, args):
        super(LitModel, self).__init__()
        assert hasattr(self, "dataset")
        self.args = args
        self.h, self.w = self.dataset.h, self.dataset.w
        self.logdir = os.path.join(args.basedir, args.model + "_" + args.expname)
        if args.debug:
            self.logdir += "_debug"
        self.i_train, self.i_val, self.i_test = self.dataset.i_train, self.dataset.i_val, self.dataset.i_test
        self.val_dummy, self.test_dummy, self.pred_dummy = self.dataset.val_dummy, self.dataset.test_dummy, self.dataset.pred_dummy
        self.near, self.far = self.dataset.near, self.dataset.far,
        self.img_size = self.h * self.w
        self.intrinsics = self.dataset.intrinsics
        self.extrinsics = self.dataset.extrinsics
        self.GL = self.dataset.GL
        self.create_model()

    def on_train_start(self):
        self.logger.log_hyperparams(self.args)

    def create_model(self):
        raise NotImplemented("Implement the [create_model] function")

    def training_step(self):
        raise NotImplemented("Implement the [training_step] function")
    
    # Utils to reorganize output values from evaluation steps, 
    # i.e., validation and test step.
    def alter_cat(self, outputs_gather, key):
        if not key in outputs_gather[0].keys():
            return None
        if torch.cuda.device_count() == 1 and not self.args.tpu:
            return torch.cat([out[key] for out in outputs_gather])
        dim = outputs_gather[0][key].shape[-1] if outputs_gather[0][key].dim() == 3 else 1 
        ret = torch.cat([out[key].transpose(1, 0).reshape(-1, dim) for out in outputs_gather]) 
        return ret

    # Gather the outputs into the ordinary device
    # and remove the dummy values for proper evaluation.
    def gather_results(self, outputs, dummy_num):
        outputs_gather = self.all_gather(outputs)
        rgbs = self.alter_cat(outputs_gather, "rgb")
        target = self.alter_cat(outputs_gather, "target")
        depths = self.alter_cat(outputs_gather, "depth")
        if dummy_num != 0:
            rgbs, depths = rgbs[:-dummy_num], depths[:-dummy_num]
            if target is not None:
                target = target[:-dummy_num]
        return rgbs, target, depths
