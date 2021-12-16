from torch.utils import data
import config
import os

from utils.select_option import select_model, select_dataloader

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

if __name__ == "__main__":

    args = config.config_parser()
    basedir = args.basedir
    expname = args.model + "_" + args.expname
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
    ) if not args.tpu else pl_loggers.TestTubeLogger(
        name=expname, savedir=logdir,
    )

    seed_everything(args.seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    model_best_checkpoint = ModelCheckpoint(
        monitor="val_psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
    )

    trainer = Trainer(
        logger=wandb_logger if args.train else None, 
        log_every_n_steps=args.i_print,
        devices=n_gpus,
        max_steps=args.max_steps,
        accelerator="gpu" if not args.tpu else "tpu",
        tpu_cores=args.tpu_num if args.tpu else None,
        replace_sampler_ddp=False,
        deterministic=True,
        strategy=DDPPlugin(find_unused_parameters=False) if not args.tpu else None,
        check_val_every_n_epoch=1,
        precision=32, 
        num_sanity_val_steps=-1 if args.debug else 0,
        callbacks=[lr_monitor, model_best_checkpoint],
    )
    dataloader = select_dataloader(args)
    info = dataloader.get_info()
    info["logdir"] = logdir
    model_fn = select_model(args)
    model = model_fn(args, info)

    if args.train:
        trainer.fit(
            model, dataloader.train_dataloader(), dataloader.val_dataloader()
        )
    if args.eval:
        best_path = os.path.join(logdir, "best.ckpt")
        trainer.test(model, dataloader.test_dataloader(), ckpt_path=best_path)

    if args.bake:
        pass
