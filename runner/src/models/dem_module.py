from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchdyn.core import NeuralODE
from torchmetrics import MeanMetric

from src.energies.base_energy_function import BaseEnergyFunction
from src.utils.logging_utils import fig_to_image

from .components.clipper import Clipper
from .components.cnf import CNF
from .components.distribution_distances import compute_distribution_distances
from .components.ema import EMAWrapper
from .components.lambda_weighter import BaseLambdaWeighter
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.scaling_wrapper import ScalingWrapper
from .components.score_estimator import estimate_grad_Rt
from .components.score_scaler import BaseScoreScaler
from .components.sde_integration import integrate_sde
from .components.sdes import PIS_SDE, VEReverseSDE

from .components.mlp import TimeConder

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
    """
    Gets the wandb logger if it is the
    list of loggers otherwise returns None.
    """
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        energy_function: BaseEnergyFunction,
        noise_schedule: BaseNoiseSchedule,
        lambda_weighter: BaseLambdaWeighter,
        buffer: PrioritisedReplayBuffer,
        num_init_samples: int,
        num_estimator_mc_samples: int,
        num_samples_to_generate_per_epoch: int,
        num_samples_to_sample_from_buffer: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        nll_with_cfm: bool,
        nll_with_dem: bool,
        cfm_sigma: float,
        cfm_prior_std: float,
        compile: bool,
        prioritize_cfm_training_samples: bool = False,
        input_scaling_factor: Optional[float] = None,
        output_scaling_factor: Optional[float] = None,
        clipper: Optional[Clipper] = None,
        score_scaler: Optional[BaseScoreScaler] = None,
        partial_prior=None,
        clipper_gen: Optional[Clipper] = None,
        diffusion_scale=1.0,
        cfm_loss_weight=1.0,
        pis_scale=1.0,
        use_ema=False,
        debug_use_train_data=False,
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
        if use_ema:
            self.net = EMAWrapper(self.net)
            self.cfm_net = EMAWrapper(self.cfm_net)
        if input_scaling_factor is not None or output_scaling_factor is not None:
            self.net = ScalingWrapper(
                self.net, input_scaling_factor, output_scaling_factor
            )

            self.cfm_net = ScalingWrapper(
                self.cfm_net, input_scaling_factor, output_scaling_factor
            )

        self.score_scaler = None
        if score_scaler is not None:
            self.score_scaler = self.hparams.score_scaler(noise_schedule)

            self.net = self.score_scaler.wrap_model_for_unscaling(self.net)
            self.cfm_net = self.score_scaler.wrap_model_for_unscaling(self.cfm_net)

        self.dem_cnf = CNF(self.net, is_diffusion=True)
        self.cfm_cnf = CNF(self.cfm_net, is_diffusion=False)

        self.nll_with_cfm = nll_with_cfm
        self.nll_with_dem = nll_with_dem
        self.cfm_prior_std = cfm_prior_std
        self.conditional_flow_matcher = ExactOptimalTransportConditionalFlowMatcher(
            sigma=cfm_sigma
        )
        # self.conditional_flow_matcher = ConditionalFlowMatcher(sigma=cfm_sigma)

        self.energy_function = energy_function
        self.noise_schedule = noise_schedule
        self.buffer = buffer
        self.dim = self.energy_function.dimensionality

        self.reverse_sde = VEReverseSDE(self.net, self.noise_schedule)

        self.clipper = clipper
        self.clipped_grad_fxn = self.clipper.wrap_grad_fxn(estimate_grad_Rt)

        self.dem_train_loss = MeanMetric()
        self.cfm_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_nll_logdetjac = MeanMetric()
        self.test_nll_logdetjac = MeanMetric()
        self.val_nll_log_p_1 = MeanMetric()
        self.test_nll_log_p_1 = MeanMetric()
        self.val_nll = MeanMetric()
        self.test_nll = MeanMetric()
        self.val_nfe = MeanMetric()
        self.test_nfe = MeanMetric()

        self.val_dem_nll_logdetjac = MeanMetric()
        self.test_dem_nll_logdetjac = MeanMetric()
        self.val_dem_nll_log_p_1 = MeanMetric()
        self.test_dem_nll_log_p_1 = MeanMetric()
        self.val_dem_nll = MeanMetric()
        self.test_dem_nll = MeanMetric()
        self.val_dem_nfe = MeanMetric()
        self.test_dem_nfe = MeanMetric()
        self.val_dem_logz = MeanMetric()
        self.val_logz = MeanMetric()
        self.test_dem_logz = MeanMetric()
        self.test_logz = MeanMetric()

        self.val_buffer_nll_logdetjac = MeanMetric()
        self.val_buffer_nll_log_p_1 = MeanMetric()
        self.val_buffer_nll = MeanMetric()
        self.val_buffer_nfe = MeanMetric()
        self.val_buffer_logz = MeanMetric()
        self.test_buffer_nll_logdetjac = MeanMetric()
        self.test_buffer_nll_log_p_1 = MeanMetric()
        self.test_buffer_nll = MeanMetric()
        self.test_buffer_nfe = MeanMetric()
        self.test_buffer_logz = MeanMetric()

        self.num_init_samples = num_init_samples
        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_samples_to_generate_per_epoch = num_samples_to_generate_per_epoch
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer
        self.num_integration_steps = num_integration_steps

        self.prioritize_cfm_training_samples = prioritize_cfm_training_samples
        self.lambda_weighter = self.hparams.lambda_weighter(self.noise_schedule)

        self.last_samples = None
        self.last_energies = None
        self.eval_step_outputs = []

        self.partial_prior = partial_prior

        self.clipper_gen = clipper_gen

        self.diffusion_scale = diffusion_scale
        self.pis_scale = pis_scale

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(t, x)

    def get_cfm_loss(self, samples: torch.Tensor) -> torch.Tensor:
        x0 = (
            torch.randn(
                self.num_samples_to_sample_from_buffer,
                self.energy_function.dimensionality,
                device=self.device,
            )
            * self.cfm_prior_std
        )
        x1 = self.energy_function.unnormalize(samples)

        t, xt, ut = self.conditional_flow_matcher.sample_location_and_conditional_flow(
            x0, x1
        )

        vt = self.cfm_net(t, xt)
        return (vt - ut).pow(2).mean(dim=-1)
        # / (
        #    self.energy_function.data_normalization_factor**2
        # )

    def should_train_cfm(self, batch_idx: int) -> bool:
        return self.nll_with_cfm

    def get_loss(self, times: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        estimated_score = estimate_grad_Rt(
            times,
            samples,
            self.energy_function,
            self.noise_schedule,
            num_mc_samples=self.num_estimator_mc_samples,
        )

        if self.clipper is not None and self.clipper.should_clip_scores:
            estimated_score = self.clipper.clip_scores(estimated_score)

        if self.score_scaler is not None:
            estimated_score = self.score_scaler.scale_target_score(
                estimated_score, times
            )

        predicted_score = self.forward(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)

        return self.lambda_weighter(times) * error_norms

    def training_step(self, batch, batch_idx):
        loss = 0.0
        if not self.hparams.debug_use_train_data:
            iter_samples, _, _ = self.buffer.sample(
                self.num_samples_to_sample_from_buffer
            )

            times = torch.rand(
                (self.num_samples_to_sample_from_buffer,), device=iter_samples.device
            )

            noised_samples = iter_samples + (
                torch.randn_like(iter_samples)
                * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
            )

            dem_loss = self.get_loss(times, noised_samples)
            self.log_dict(
                t_stratified_loss(
                    times, dem_loss, loss_name="train/stratified/dem_loss"
                )
            )
            dem_loss = dem_loss.mean()
            loss = loss + dem_loss

            # update and log metrics
            self.dem_train_loss(dem_loss)
            self.log(
                "train/dem_loss",
                self.dem_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.should_train_cfm(batch_idx):
            if self.hparams.debug_use_train_data:
                cfm_samples = self.energy_function.sample_train_set(
                    self.num_samples_to_sample_from_buffer
                )
                times = torch.rand(
                    (self.num_samples_to_sample_from_buffer,),
                    device=cfm_samples.device
                )
            else:
                cfm_samples, _, _ = self.buffer.sample(
                    self.num_samples_to_sample_from_buffer,
                    prioritize=self.prioritize_cfm_training_samples,
                )

            cfm_loss = self.get_cfm_loss(cfm_samples)
            self.log_dict(
                t_stratified_loss(
                    times, cfm_loss, loss_name="train/stratified/cfm_loss"
                )
            )
            cfm_loss = cfm_loss.mean()
            self.cfm_train_loss(cfm_loss)
            self.log(
                "train/cfm_loss",
                self.cfm_train_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            loss = loss + self.hparams.cfm_loss_weight * cfm_loss
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.hparams.use_ema:
            self.net.update_ema()
            if self.should_train_cfm(batch_idx):
                self.cfm_net.update_ema()

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch

        samples = self.prior.sample(num_samples)

        return self.integrate(
            reverse_sde=reverse_sde,
            samples=samples,
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
        )

    def integrate(
        self,
        reverse_sde: VEReverseSDE = None,
        samples: torch.Tensor = None,
        reverse_time=True,
        return_full_trajectory=False,
        diffusion_scale=1.0,
        no_grad=True,
    ) -> torch.Tensor:
        trajectory = integrate_sde(
            reverse_sde or self.reverse_sde,
            samples,
            self.num_integration_steps + 1,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
        )
        if return_full_trajectory:
            return trajectory
        return trajectory[-1]

    def compute_nll(
        self,
        cnf,
        prior,
        samples: torch.Tensor,
        num_integration_steps=1,
        method="dopri5",
    ):
        aug_samples = torch.cat(
            [samples, torch.zeros(samples.shape[0], 1, device=samples.device)], dim=-1
        )

        aug_output = cnf.integrate(aug_samples, num_integration_steps=1, method=method)[
            -1
        ]
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
            self.last_samples = self.generate_samples(
                diffusion_scale=self.diffusion_scale
            )
            self.last_energies = self.energy_function(self.last_samples)

        self.buffer.add(self.last_samples, self.last_energies)

    def compute_and_log_nll(self, cnf, prior, samples, prefix, name):
        cnf.nfe = 0.0
        nll, forwards_samples, logdetjac, log_p_1 = self.compute_nll(
            cnf, prior, samples
        )
        # Normalize, this seems super weird, but is the right thing to do -- AT
        logz = self.energy_function(self.energy_function.normalize(samples)) + nll
        nfe_metric = getattr(self, f"{prefix}_{name}nfe")
        nll_metric = getattr(self, f"{prefix}_{name}nll")
        logdetjac_metric = getattr(self, f"{prefix}_{name}nll_logdetjac")
        log_p_1_metric = getattr(self, f"{prefix}_{name}nll_log_p_1")
        logz_metric = getattr(self, f"{prefix}_{name}logz")
        nfe_metric.update(cnf.nfe)
        nll_metric.update(nll)
        logdetjac_metric.update(logdetjac)
        log_p_1_metric.update(log_p_1)
        logz_metric.update(logz)

        self.log_dict(
            {
                f"{prefix}/{name}_nfe": nfe_metric,
                f"{prefix}/{name}nll_logdetjac": logdetjac_metric,
                f"{prefix}/{name}nll_log_p_1": log_p_1_metric,
                f"{prefix}/{name}logz": logz_metric,
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

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        times = torch.rand(
            (self.num_samples_to_generate_per_epoch,), device=batch.device
        )

        batch = self.energy_function.sample_test_set(
            self.num_samples_to_generate_per_epoch
        )

        noised_batch = batch + (
            torch.randn_like(batch) * self.noise_schedule.h(times).sqrt().unsqueeze(-1)
        )

        loss = self.get_loss(times, noised_batch).mean(-1)

        # generate samples noise --> data if needed
        backwards_samples = self.last_samples
        if backwards_samples is None:
            backwards_samples = self.generate_samples(num_samples=len(batch))

        # update and log metrics
        loss_metric = self.val_loss if prefix == "val" else self.test_loss
        loss_metric(loss)

        self.log(
            f"{prefix}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True
        )

        to_log = {
            "data_0": batch,
            "gen_0": backwards_samples,
        }
        if self.nll_with_dem:
            forwards_samples = self.compute_and_log_nll(
                self.dem_cnf, self.prior, batch, prefix, "dem_"
            )
            to_log["gen_1_dem"] = forwards_samples
        if self.nll_with_cfm:
            forwards_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, batch, prefix, ""
            )
            to_log["gen_1_cfm"] = forwards_samples
            iter_samples, _, _ = self.buffer.sample(
                self.num_samples_to_generate_per_epoch
            )
            forwards_samples = self.compute_and_log_nll(
                self.cfm_cnf, self.cfm_prior, iter_samples, prefix, "buffer_"
            )

        self.eval_step_outputs.append(to_log)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("val", batch, batch_idx)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.eval_step("test", batch, batch_idx)

    def generate_cfm_samples(self):
        def reverse_wrapper(model):
            def fxn(t, x, args=None):
                if t.ndim == 0:
                    t = t.unsqueeze(0)

                return model(t.repeat(len(x)), x)

            return fxn

        node = NeuralODE(
            reverse_wrapper(self.cfm_net),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        with torch.no_grad():
            shape = (
                self.num_samples_to_generate_per_epoch,
                self.energy_function.dimensionality,
            )

            noise = torch.randn(shape, device=self.device) * self.cfm_prior_std
            traj = node.trajectory(
                noise,
                t_span=torch.linspace(0, 1, 2, device=self.device),
            )

            return traj[-1]

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
        wandb_logger = get_wandb_logger(self.loggers)
        # convert to dict of tensors assumes [batch, ...]
        outputs = {
            k: torch.cat([dic[k] for dic in self.eval_step_outputs], dim=0)
            for k in self.eval_step_outputs[0]
        }

        if self.energy_function.dimensionality == 2:
            if self.nll_with_cfm:
                self.scatter_prior(prefix + "/cfm", outputs["gen_1_cfm"])
            if self.nll_with_dem:
                self.scatter_prior(prefix + "/dem", outputs["gen_1_dem"])

        unprioritized_buffer_samples, cfm_samples = None, None
        if self.nll_with_cfm:
            unprioritized_buffer_samples, _, _ = self.buffer.sample(
                self.num_samples_to_generate_per_epoch,
                prioritize=self.prioritize_cfm_training_samples,
            )

            cfm_samples = self.generate_cfm_samples()

        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            unprioritized_buffer_samples,
            cfm_samples,
            self.buffer,
            wandb_logger,
        )

        # pad with time dimension 1
        names, dists = compute_distribution_distances(
            outputs["gen_0"][:, None], outputs["data_0"][:, None]
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

        def _grad_fxn(t, x):
            return self.clipped_grad_fxn(
                t,
                x,
                self.energy_function,
                self.noise_schedule,
                self.num_estimator_mc_samples,
            )

        reverse_sde = VEReverseSDE(_grad_fxn, self.noise_schedule)

        self.prior = self.partial_prior(
            device=self.device, scale=self.noise_schedule.h(1) ** 0.5
        )

        init_states = self.generate_samples(
            reverse_sde, self.num_init_samples, diffusion_scale=self.diffusion_scale
        )
        init_energies = self.energy_function(init_states)

        self.buffer.add(init_states, init_energies)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            self.cfm_net = torch.compile(self.cfm_net)

        if self.nll_with_cfm:
            self.cfm_prior = self.partial_prior(
                device=self.device, scale=self.cfm_prior_std
            )

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


class PISLitModule(DEMLitModule):
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        aug_prior_samples = torch.zeros(
            self.num_samples_to_generate_per_epoch, self.dim + 1, device=self.device
        )

        aug_output = self.integrate(
            self.pis_sde,
            aug_prior_samples,
            return_full_trajectory=True,
            no_grad=False,
            reverse_time=False
        )[-1]
        x_1, quad_reg = aug_output[..., :-1], aug_output[..., -1]
        prior_ll = self.prior.log_prob(x_1).mean() / self.dim
        sample_ll = self.energy_function(x_1).mean() / self.dim
        term_loss = prior_ll - sample_ll
        quad_reg = (quad_reg).mean() / self.dim
        loss = term_loss + quad_reg
        self.log_dict(
            {
                "train/reg_loss": quad_reg,
                "train/prior_ll": prior_ll,
                "train/sample_ll": sample_ll,
                "train/term_loss": term_loss,
            }
        )

        # update and log metrics
        self.pis_train_loss(loss)
        self.log(
            "train/loss",
            self.pis_train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale=1.0,
    ) -> torch.Tensor:
        num_samples = num_samples or self.num_samples_to_generate_per_epoch
        samples = torch.zeros(
            num_samples, self.dim + 1, device=self.device
        )

        return self.integrate(
            reverse_sde=self.pis_sde,
            samples=samples,
            reverse_time=False,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale,
        )[..., :-1]

    def setup(self, stage: str) -> None:
        self.tcond = TimeConder(64, 1, 3)
        self.prior = self.partial_prior(
            device=self.device, scale=self.pis_scale
        )
        self.tcond = TimeConder(64, 1, 3).to(self.device)
        self.pis_sde = PIS_SDE(self.net, self.tcond, self.pis_scale, self.energy_function).to(self.device)
        self.pis_train_loss = MeanMetric()
        # torch.nn.init.zeros_(self.net.joint_mlp[-1].weight.data)
        # torch.nn.init.zeros_(self.net.joint_mlp[-1].bias.data)
        super().setup(stage)

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
