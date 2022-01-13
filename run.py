import config
import os
import yaml

from utils.select_option import select_model, select_callback, select_config

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import logging

if __name__ == "__main__":

    logging.getLogger("lightning").setLevel(logging.ERROR)
    args, parser = config.config_parser()
    args.datadir = args.datadir.rstrip("/")
    if args.config is None:
        args.config = select_config(args)

    with open(args.config, "r") as fp:
        config_file = yaml.load(fp, Loader=yaml.FullLoader)

    model_name = config_file["model"]
    dataset_type = config_file["dataset_type"]
    model_fn = select_model(model_name, dataset_type)
    args = model_fn.add_model_specific_args(parser)
    args.__dict__.update(config_file)

    args.datadir = args.datadir.rstrip("/")
    basedir = args.basedir
    if args.expname is None:
        args.expname = args.datadir.split("/")[-1]
    args.expname = args.model + "_" + args.expname + args.postfix
    if args.debug:
        args.expname += "_debug"
    logdir = os.path.join(basedir, args.expname)

    n_gpus = torch.cuda.device_count()

    os.makedirs(logdir, exist_ok=True)

    if args.train:
        f = os.path.join(logdir, "args.txt")
        with open(f, "w") as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write("{} = {}\n".format(arg, attr))

    wandb_logger = pl_loggers.WandbLogger(
        name=args.expname, entity="postech_cvlab",
        project="idg"
        ) if not args.tpu else pl_loggers.TensorBoardLogger(
            save_dir=logdir, name=args.expname)

    seed_everything(args.seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val_psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
    )

    callbacks = [lr_monitor, model_checkpoint]
    callbacks = select_callback(callbacks, model_name, args)

    trainer = Trainer(
        logger=wandb_logger if args.train else None,
        log_every_n_steps=args.i_print,
        devices=n_gpus,
        max_steps=args.max_steps,
        accelerator="gpu" if not args.tpu else "tpu",
        tpu_cores=args.tpu_num if args.tpu else None,
        replace_sampler_ddp=False,
        strategy=DDPPlugin(find_unused_parameters=False) \
            if n_gpus > 1 and not args.tpu else None,
        check_val_every_n_epoch=1,
        precision=32,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )

    model = model_fn(args)

    if args.train:
        trainer.fit(model)
    if args.eval:
        best_path = os.path.join(logdir, "best.ckpt")
        trainer.test(model, ckpt_path=best_path)
    if args.render:
        best_path = os.path.join(logdir, "best.ckpt")
        trainer.predict(model, ckpt_path=best_path)
    if args.bake:
        pass
