from torch.utils import data
import config
import os

from utils.select_option import select_trainer, select_dataloader

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

if __name__ == "__main__":

    args = config.config_parser()
    basedir = args.basedir
    expname = args.expname
    logdir = os.path.join(basedir, expname)

    n_gpus = torch.cuda.device_count()

    os.makedirs(logdir, exist_ok=True)
    
    f = os.path.join(logdir, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    f = os.path.join(logdir, "config.txt")
    with open(f, "w") as file:
        file.write(open(args.config, "r").read())

    wandb_logger = pl_loggers.WandbLogger(
        name=expname, entity="postech_cvlab", project=args.model
    )

    seed_everything(args.seed, workers=True)
    dataloader = select_dataloader(args)
    info = dataloader.get_info()
    model = select_trainer(args, info)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        dirpath=logdir, filename="ckpt", save_last=True
    )

    trainer = Trainer(
        logger=wandb_logger, 
        log_every_n_steps=args.i_print,
        devices=n_gpus,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        deterministic=True,
        precision=16, 
        num_sanity_val_steps=0,
        callbacks=[lr_monitor, model_checkpoint], 
    )

    if not args.eval:
        trainer.fit(model, dataloader.train_dataloader())
    else:
        trainer.test(model, dataloader.test_dataloader())