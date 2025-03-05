import time
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from src.energies.base_energy_function import BaseEnergyFunction
from src.utils.data_utils import remove_mean
from src.utils.logging_utils import fig_to_image
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchmetrics import MeanMetric

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt, wrap_for_richardsons
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import VEReverseSDE


def t_stratified_loss(batch_t, batch_loss, num_bins=5, loss_name=None):
    """Stratify loss by binning t."""
    flat_losses = batch_loss.flatten().detach().cpu().numpy()
    flat_t = batch_t.flatten().detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


class CFMLitModule(LightningModule):
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        cfm_sigma: float,
        cfm_prior_std: float,
        partial_prior,
        use_otcfm: bool,
        nll_integration_method: str,
        compile: bool,
        use_ema=False,
        use_exact_likelihood=True,
        tol=1e-5,
        version=1,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param buffer: Buffer of sampled objects
        """
        super().__init__()
        # Seems to slow things down
        # torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net(energy_function=energy_function)
        self.cfm_net = net(energy_function=energy_function)
        self.partial_prior = partial_prior

        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)

        self.cfm_cnf = CNF(
            self.cfm_net,
            is_diffusion=False,
            use_exact_likelihood=use_exact_likelihood,
            method=nll_integration_method,
            num_steps=num_integration_steps,
            atol=tol,
            rtol=tol,
        )

        self.cfm_prior_std = cfm_prior_std

        flow_matcher = ConditionalFlowMatcher
        if use_otcfm:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher

        self.cfm_sigma = cfm_sigma
        self.conditional_flow_matcher = flow_matcher(sigma=cfm_sigma)

        self.nll_integration_method = nll_integration_method

        self.energy_function = energy_function
        self.dim = self.energy_function.dimensionality

        self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule)

        self.num_integration_steps = num_integration_steps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = self.cfm_prior.sample(self.num_samples_to_sample_from_buffer)
        x1 = samples
        x1 = self.energy_function.unnormalize(x1)

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(x0, x1)

        if self.energy_function.is_molecule and self.cfm_sigma != 0:
            xt = remove_mean(
                xt, self.energy_function.n_particles, self.energy_function.n_spatial_dim
            )

        vt = self.cfm_net(t, xt)
        loss = (vt - ut).pow(2).mean(dim=-1)

        return loss

    def training_step(self, batch, batch_idx):
        x1 = self.prior.sample(batch.shape[0])
        x0 = batch
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.should_train_cfm(batch_idx):
            self.cfm_net.update_ema()

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
    ):
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )
        aug_output = cnf.integrate(aug_samples)[-1]
        x_1, logdetjac = aug_output[..., :-1], aug_output[..., -1]
        log_p_1 = prior.log_prob(x_1)
        log_p_0 = log_p_1 + logdetjac
        nll = -log_p_0
        return nll, x_1, logdetjac, log_p_1

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # self.last_samples = self.generate_samples()
        # self.last_energies = self.energy_function(self.last_samples)
        if self.clipper_gen is not None:
            reverse_sde = VEReverseSDE(
                self.clipper_gen.wrap_grad_fxn(self.net), self.noise_schedule
            )
            self.last_samples = self.generate_samples(
                reverse_sde=reverse_sde, diffusion_scale=self.diffusion_scale
            )
            self.last_energies = self.energy_function(self.last_samples)
        else:
            self.last_samples = self.generate_samples(diffusion_scale=self.diffusion_scale)
            self.last_energies = self.energy_function(self.last_samples)

        self.buffer.add(self.last_samples, self.last_energies)

        self._log_energy_w2()

    def _log_energy_w2(self):
        if len(self.buffer) < self.eval_batch_size:
            return

        val_set = self.energy_function.sample_val_set(self.eval_batch_size)
        val_energies = self.energy_function(self.energy_function.normalize(val_set))

        _, generated_energies = self.buffer.get_last_n_inserted(self.eval_batch_size)

        energy_w2 = pot.emd2_1d(val_energies.cpu().numpy(), generated_energies.cpu().numpy())

        self.log(
            f"val/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_log_z(self, cnf, prior, samples, prefix, name):
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(cnf, prior, samples)
        # this is stupid we should fix the normalization in the energy function
        logz = self.energy_function(self.energy_function.normalize(samples)) + nll
        logz_metric = getattr(self, f"{prefix}_{name}logz")
        logz_metric.update(logz)
        self.log(
            f"{prefix}/{name}logz",
            logz_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_and_log_nll(self, cnf, prior, samples, prefix, name):
        cnf.nfe = 0.0
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(cnf, prior, samples)
        nfe_metric = getattr(self, f"{prefix}_{name}nfe")
        nll_metric = getattr(self, f"{prefix}_{name}nll")
        logdetjac_metric = getattr(self, f"{prefix}_{name}nll_logdetjac")
        log_p_1_metric = getattr(self, f"{prefix}_{name}nll_log_p_1")
        nfe_metric.update(cnf.nfe)
        nll_metric.update(nll)
        logdetjac_metric.update(logdetjac)
        log_p_1_metric.update(log_p_1)

        self.log_dict(
            {
                f"{prefix}/{name}_nfe": nfe_metric,
                f"{prefix}/{name}nll_logdetjac": logdetjac_metric,
                f"{prefix}/{name}nll_log_p_1": log_p_1_metric,
                # f"{prefix}/{name}logz": logz_metric,
            },
            on_epoch=True,
        )
        self.log(
            f"{prefix}/{name}nll",
            nll_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return forwards_samples

    def get_loss(self, x0, x1):
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        vt = self(t, xt)
        loss = (vt - ut).pow(2).mean()
        return loss

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x1 = self.cfm_prior.sample(self.eval_batch_size)
        x0 = batch
        loss = self.get_loss(x0, x1)
        # generate samples noise --> data if needed
        backwards_samples = self.generate_samples(num_samples=self.eval_batch_size)
        self.eval_step_outputs.append({"gen_0": backwards_samples})

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)
        self.log(f"{prefix}/loss", loss_metric, on_step=True, on_epoch=True, prog_bar=True)
        forwards_samples = self.compute_and_log_nll(
            self.cfm_cnf, self.cfm_prior, batch, prefix, ""
        )
        backwards_samples = self.cfm_cnf.generate(x1)[-1]
        to_log = {"data_0": batch, "gen_0": backwards_samples, "gen_1_cfm": forwards_samples}
        iter_samples, _, _ = self.buffer.sample(self.eval_batch_size)

        # backwards_samples = self.generate_cfm_samples(self.eval_batch_size)
        self.compute_log_z(self.cfm_cnf, self.cfm_prior, backwards_samples, prefix, "")

        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        batch = self.energy_function.sample_val_set(self.eval_batch_size)
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        batch = self.energy_function.sample_test_set(self.eval_batch_size)
        self.eval_step("test", batch, batch_idx)

    def scatter_prior(self, prefix, outputs):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, ax = plt.subplots()
        n_samples = outputs.shape[0]
        ax.scatter(*outputs.detach().cpu().T, label="Generated prior")
        ax.scatter(
            *self.prior.sample(n_samples).cpu().T,
            label="True prior",
            alpha=0.5,
        )
        ax.legend()
        wandb_logger.log_image(f"{prefix}/generated_prior", [fig_to_image(fig)])

    def eval_epoch_end(self, prefix: str):
        # wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        if self.energy_function.dimensionality == 2:
            self.scatter_prior(prefix + "/cfm/prior", outputs["gen_1_cfm"])
            self.scatter_prior(prefix + "/cfm/gen", outputs["gen_0"])
            self.scatter_prior(prefix + "/cfm/data", outputs["data_0"])

            # cfm_samples = self.generate_cfm_samples(self.eval_batch_size)
        #            cfm_samples = self.cfm_cnf.generate(
        #                self.cfm_prior.sample(self.eval_batch_size),
        #            )[-1]

        if "data_0" in outputs:
            # pad with time dimension 1
            names, dists = compute_distribution_distances(
                self.energy_function.unnormalize(outputs["gen_0"])[:, None],
                outputs["data_0"][:, None],
            )
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            self.log_dict(d, sync_dist=True)

        self.eval_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self.eval_epoch_end("test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm:
            self.cfm_prior = self.partial_prior(device=self.device, scale=self.cfm_prior_std)

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
