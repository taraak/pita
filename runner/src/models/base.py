import logging
from typing import Any, Dict

import torch
from lightning import LightningModule

logger = logging.getLogger(__name__)


class BaseLightningModule(LightningModule):
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            logger.info("Skipping validation epoch end during sanity check.")
            return
        if self.trainer.global_step == 0:
            logger.info("Skipping validation epoch end during first step.")
            return
        self.eval_epoch_end("val")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.hparams.lr_scheduler_update_frequency,
                },
            }
        return {"optimizer": optimizer}
