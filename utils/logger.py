import os
import time

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from wandb.wandb_run import Run

MAX_RETRY = 100


class RetryingWandbLogger(WandbLogger):
    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            print("Initializing wandb")
            for i in range(MAX_RETRY):
                try:
                    self._experiment = wandb.init(
                        **self._wandb_init,
                    )
                    break
                except (
                    TimeoutError,
                    ConnectionError,
                    wandb.errors.UsageError,
                    wandb.errors.CommError,
                ) as e:
                    print(f"Error {e}. Retrying in 5 sec")
                    time.sleep(5)

            # save checkpoints in wandb dir to upload on W&B servers
            if self._log_model:
                self._save_dir = self._experiment.dir
        return self._experiment