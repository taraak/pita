from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from src.energies.base_energy_function import BaseEnergyFunction
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components.clipper import Clipper
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.score_estimator import estimate_grad_Rt
from .components.sde_integration import integrate_sde
from .components.sdes import VEReverseSDE


class DEMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        cfm_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_samples_per_training_step: int,
        num_to_samples_to_generate_per_epoch: int,
        num_integration_steps: int,
        compile: bool,
        clipper: Optional[Clipper] = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.cfm_net = cfm_net

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer

        self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule)

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(estimate_grad_Rt)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.num_init_samples = num_init_samples
        self.num_samples_per_training_step = num_samples_per_training_step
        self.num_to_samples_to_generate_per_epoch = num_to_samples_to_generate_per_epoch
        self.num_integration_steps = num_integration_steps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self) -> torch.Tensor:
        raise NotImplementedError

    def should_train_cfm(self, batch_idx: int) -> bool:
        return False

    def get_loss(self, times: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        estimated_score = estimate_grad_Rt(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_samples_per_training_step,
        )

        if self.clipper is not None and self.clipper.should_clip_scores:
            estimated_score = self.clipper.clip_scores(estimated_score)

        predicted_score = self.forward(times, samples)

        return (
            torch.linalg.vector_norm(predicted_score - estimated_score, dim=-1)
            .pow(2)
            .mean()
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        iter_samples, _, _ = self.buffer.sample(self.num_samples_per_training_step)

        times = torch.rand(
            (self.num_samples_per_training_step,), device=iter_samples.device
        )

        loss = self.get_loss(times, iter_samples)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        if self.should_train_cfm(batch_idx):
            loss = loss + self.get_cfm_loss(iter_samples, times)

        # return loss or backpropagation will fail
        return loss

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: int = None
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_to_samples_to_generate_per_epoch
        noise = torch.randn(
            (num_samples, self.energy_function.dimensionality),
            device=self.device
        ) * (self.noise_schedule.h(1) ** 0.5)

        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            noise,
            self.num_integration_steps + 1
        )

        return trajectory[-1]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        samples = self.generate_samples()
        sample_energies = self.energy_function(samples)

        self.buffer.add(samples, sample_energies)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        times = torch.rand((self.num_samples_per_training_step,), device=batch.device)
        batch = self.energy_function.sample_test_set(self.num_samples_per_training_step)
        loss = self.get_loss(times, batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # TODO: Add all of our metrics here and evaluate them!
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        def _grad_fxn(t, x, noise_schedule):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                noise_schedule,
                self.num_samples_per_training_step
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        init_states = self.generate_samples(reverse_sde, self.num_init_samples)
        init_energies = self.energy_function(init_states)

        self.buffer.add(init_states, init_energies)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

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
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def test_step(self, batch, batch_idx):
        pass

if __name__ == "__main__":
    _ = DEMLitModule(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
