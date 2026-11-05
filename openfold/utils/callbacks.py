# Modified by Adaptive Biotechnologies
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
except ImportError:
    WandbLogger = None
    wandb = None

class EarlyStoppingVerbose(EarlyStopping):
    """
        The default EarlyStopping callback's verbose mode is too verbose.
        This class outputs a message only when it's getting ready to stop. 
    """
    def _evalute_stopping_criteria(self, *args, **kwargs):
        should_stop, reason = super()._evalute_stopping_criteria(*args, **kwargs)
        if(should_stop):
            rank_zero_info(f"{reason}\n")

        return should_stop, reason

class ModelCheckpointWandb(ModelCheckpoint):
    """
    Upload to wandb every time a local checkpoint is created.
    If wandb is not running, it behaves the same way as ModelCheckpoint.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_logger = None

    def set_wandb_logger(self, wandb_logger: WandbLogger):
        """
        Set WandbLogger so it can be used to upload checkpoints to wandb. 
        """
        self.wandb_logger = wandb_logger

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str):
        # parent class saves a checkpoint locally in filepath
        super()._save_checkpoint(trainer, filepath)

        # if wandb is running, upload checkpoint to wandb server
        if (
            self.wandb_logger
            and self.wandb_logger.experiment
            and trainer.global_rank == 0
        ):
            artifact = wandb.Artifact(
                name=f"{self.wandb_logger.experiment.name}_checkpoint",
                type="checkpoint",
                metadata={
                    "epoch": trainer.current_epoch,
                    "step": trainer.global_step,
                    "val_loss": trainer.callback_metrics.get("val/loss", None)
                },
            )
            artifact.add_dir(filepath)
            self.wandb_logger.experiment.log_artifact(artifact)
