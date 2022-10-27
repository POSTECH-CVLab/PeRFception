import argparse
import logging
import os
import shutil
from typing import *

import gin
import pytorch_lightning.loggers as pl_loggers
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins import DDPPlugin

from utils.logger import RetryingWandbLogger
from utils.select_option import select_callback, select_dataset, select_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


@gin.configurable()
def run(
    resume_training: bool = False,
    ckpt_path: Optional[str] = None,
    datadir: Optional[str] = None,
    logbase: Optional[str] = None,
    scene_name: Optional[str] = None,
    model_name: Optional[str] = None,
    proj_name: Optional[str] = "perf-co3d-v2",
    dataset_name: Optional[str] = None,
    postfix: Optional[str] = None,
    entity: Optional[str] = None,
    # Optimization
    max_steps: int = 200000,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 5,
    # Run Mode
    run_train: bool = True,
    run_eval: bool = True,
    run_render: bool = False,
    accelerator: str = "gpu",
    num_gpus: Optional[int] = 1,
    num_tpus: Optional[int] = None,
    num_sanity_val_steps: int = 0,
    seed: int = 777,
    debug: bool = False,
    save_last_only: bool = False,
    check_val_every_n_epoch: int = 1,
):

    logging.getLogger("lightning").setLevel(logging.ERROR)
    datadir = datadir.rstrip("/")

    if scene_name is None and dataset_name == "co3d":
        scene_name = "349_36520_66801"

    if scene_name is None and dataset_name == "scannet":
        scene_name = "scene0000_00"

    exp_name = model_name + "_" + dataset_name + "_" + scene_name
    if postfix is not None:
        exp_name += "_" + str(postfix)
    if debug:
        exp_name += "_debug"

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    os.makedirs(logbase, exist_ok=True)
    logdir = os.path.join(logbase, exp_name)

    os.makedirs(logdir, exist_ok=True)

    # WANDB fails when using TPUs
    wandb_logger = (
        RetryingWandbLogger(
            name=exp_name,
            entity=entity,
            project=model_name if proj_name is None else proj_name,
        )
        if accelerator == "gpu"
        else pl_loggers.TensorBoardLogger(save_dir=logdir, name=exp_name)
    )

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val/psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
        save_last=save_last_only,
    )
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = [lr_monitor, model_checkpoint, tqdm_progrss]
    callbacks += select_callback(model_name)

    trainer = Trainer(
        logger=wandb_logger if run_train or run_render else None,
        log_every_n_steps=log_every_n_steps,
        devices=num_gpus,
        max_steps=max_steps,
        replace_sampler_ddp=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=precision,
        accelerator="gpu",
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
    )

    if resume_training:
        if ckpt_path is None:
            ckpt_path = f"{logdir}/last.ckpt"

    data_module = select_dataset(
        dataset_name=dataset_name,
        scene_name=scene_name,
        datadir=datadir,
        accelerator="gpu",
        num_gpus=num_gpus,
        num_tpus=num_tpus,
    )
    model = select_model(model_name=model_name)
    model.logdir = logdir
    if run_train:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
        if save_last_only:
            best_ckpt = os.path.join(logdir, "best.ckpt")
            if os.path.exists(best_ckpt):
                os.remove(best_ckpt)
    ckpt_path = f"{logdir}/best.ckpt" if not save_last_only else f"{logdir}/last.ckpt"
    if run_eval:
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    if run_render:
        trainer.predict(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="gin bindings",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoints",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="scene name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="postech_cvlab",
        help="entity",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    if args.ginc is None:
        args.ginc = ["configs/co3d.gin"]

    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        resume_training=args.resume_training,
        ckpt_path=args.ckpt_path,
        scene_name=args.scene_name,
        entity=args.entity,
    )
