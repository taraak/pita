import copy
import logging
import time
from dataclasses import fields
from typing import List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import PIL
import torch
import wandb
from src.energies.base_energy_function import BaseEnergyFunction
from src.energies.components.center_of_mass import CenterOfMassTransform
from src.energies.components.rotation import Random3DRotationTransform
from src.models.base import BaseLightningModule
from src.models.components.ema import EMA
from src.models.components.energy_net import EnergyNet
from src.models.components.noise_schedules import BaseNoiseSchedule
from src.models.components.score_net import FlowNet, ScoreNet
from src.models.components.sde_integration import WeightedSDEIntegrator
from src.models.components.sdes import SDETerms, VEReverseSDE
from src.models.components.utils import (
    compiled_divergence_fn,
    get_wandb_logger,
    sample_from_tensor,
)
from src.utils.data_utils import remove_mean

from .components.clipper import Clipper
from .components.distribution_distances import energy_distances
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.score_estimator import estimate_grad_Rt, estimate_Rt

logger = logging.getLogger(__name__)

# set matmul precision to medium
torch.set_float32_matmul_precision("high")


class MyModuleDict(torch.nn.ModuleDict):
    def __setitem__(self, key, value):
        key = str(key)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        key = str(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        key = str(key)
        return super().__contains__(key)


class energyTempModule(BaseLightningModule):
    strict_loading = False

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        clipper: Clipper,
        noise_schedule: BaseNoiseSchedule,
        partial_buffer: PrioritisedReplayBuffer,
        training_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        num_samples_to_save: int,
        num_init_samples: int,
        temperatures: List[float],
        num_eval_samples: int,
        test_batch_size: int,
        inference_batch_size: int,
        start_resampling_step: int,
        end_resampling_step: int,
        resampling_interval: int,
        num_epochs_per_temp: List[int],
        num_negative_time_steps: int,
        resample_at_end: bool,
        compile: bool,
        loss_weights: dict,
        partial_prior=None,
        init_from_prior=False,
        seed=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.h_theta = self.hparams.net()
        self.score_net = ScoreNet(
            model=self.h_theta, precondition_beta=self.hparams.precondition_beta
        )
        self.energy_net = EnergyNet(
            score_net=copy.deepcopy(self.h_theta), precondition_beta=self.hparams.precondition_beta
        )
        self.score_net_forward = self.score_net.forward
        self.energy_net_forward_energy = self.energy_net.forward_energy
        self.energy_net_forward = self.energy_net.forward
        self.energy_net_denoiser_and_energy = self.energy_net.denoiser_and_energy

        self.is_molecule = self.hparams.is_molecule
        self.should_mean_free = self.hparams.should_mean_free and self.is_molecule

        if self.hparams.ema_decay > 0:
            self.score_net = EMA(self.score_net, decay=self.hparams.ema_decay)
            self.energy_net = EMA(self.energy_net, decay=self.hparams.ema_decay)

        # self.compiled_divergence = compiled_divergence_fn(self.score_net.forward)
        if self.hparams.compile:
            self.score_net_forward = torch.compile(self.score_net.forward)
            self.energy_net_forward_energy = torch.compile(self.energy_net.forward_energy)
            self.energy_net_forward = torch.compile(self.energy_net.forward)
            # Cannot compile denoiser and energy as we can't do a backward pass through it
            # https://github.com/pytorch/pytorch/issues/91469
            # self.energy_net_denoiser_and_energy = torch.compile(
            #     self.energy_net.denoiser_and_energy
            # )

        self.reverse_sde = VEReverseSDE(
            energy_net=self.energy_net,
            noise_schedule=self.hparams.noise_schedule,
            score_net=self.score_net,
            # compiled_divergence_fn=None,
            pin_energy=False,
            debias_inference=self.hparams.debias_inference,
        )
        if self.hparams.dem.num_training_epochs > 0:
            self.dem_reverse_sde = VEReverseSDE(
                noise_schedule=self.hparams.dem.noise_schedule,
                score_net=self.score_net,
                debias_inference=False,
            )
            self.weighted_sde_integrator_dem = WeightedSDEIntegrator(
                sde=self.dem_reverse_sde,
                num_integration_steps=self.hparams.num_integration_steps,
                reverse_time=True,
                time_range=1.0,
                no_grad=True,
                diffusion_scale=self.hparams.diffusion_scale,
                resampling_interval=-1,
                num_negative_time_steps=self.hparams.num_negative_time_steps,
                post_mcmc_steps=self.hparams.post_mcmc_steps,
                adaptive_mcmc=self.hparams.adaptive_mcmc,
                start_resampling_step=0,
                end_resampling_step=self.hparams.dem.num_integration_steps,
                resample_at_end=False,
                batch_size=self.hparams.inference_batch_size,
                lightning_module=self,
                should_mean_free=self.should_mean_free,
            )

        n_steps = self.hparams.num_integration_steps
        self.weighted_sde_integrator = WeightedSDEIntegrator(
            sde=self.reverse_sde,
            num_integration_steps=n_steps,
            reverse_time=True,
            time_range=1.0,
            no_grad=True,
            diffusion_scale=self.hparams.diffusion_scale,
            resampling_interval=self.hparams.resampling_interval,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
            post_mcmc_steps=self.hparams.post_mcmc_steps,
            adaptive_mcmc=self.hparams.adaptive_mcmc,
            dt_negative_time=self.hparams.dt_negative_time,
            do_langevin=self.hparams.do_langevin,
            start_resampling_step=self.hparams.start_resampling_step,
            end_resampling_step=self.hparams.end_resampling_step,
            resample_at_end=self.hparams.resample_at_end,
            batch_size=self.hparams.inference_batch_size,
            lightning_module=self,
            should_mean_free=self.should_mean_free,
            partial_annealing_factor_schedule=self.hparams.annealing_factor_schedule,
        )

        self.register_buffer("temperatures", torch.tensor(self.hparams.temperatures))
        self.register_buffer(
            "inverse_temperatures",
            torch.round(self.temperatures[0] / self.temperatures, decimals=2),
        )
        n_temps = len(self.inverse_temperatures)

        logger.debug(f"Temperatures: {self.temperatures}")
        logger.debug(f"Inverse Temperatures: {self.inverse_temperatures}")
        self.sample_buffers = MyModuleDict(
            {str(temp_index): self.hparams.partial_buffer() for temp_index in range(n_temps)}
        )
        times = torch.linspace(1, 0, self.hparams.num_integration_steps + 1)
        self.t_start = times[self.hparams.start_resampling_step]

    @property
    def active_inverse_temperature_index(self):
        to_return = int(
            np.searchsorted(self.update_temp_epoch, self.trainer.current_epoch, side="right")
        )
        return to_return

    def debug_generate_samples(
        self,
        energy_function: BaseEnergyFunction,
        num_samples: int,
        inverse_temp: Optional[float] = 1.0,
        annealing_factor: Optional[float] = 1.0,
        return_full_trajectory: Optional[bool] = False,
        weighted_sde_integrator: Optional[WeightedSDEIntegrator] = None,
    ) -> torch.Tensor:
        annealing_factor_schedule = self.hparams.annealing_factor_schedule(
            annealing_factor=annealing_factor
        )
        prior = self.hparams.partial_prior(
            scale=(
                self.hparams.noise_schedule.h(self.t_start)
                / annealing_factor_schedule.gamma(self.t_start)
            )
            ** 0.5,
            device=self.device,
        )

        prior_samples = prior.sample(num_samples)
        if weighted_sde_integrator is None:
            weighted_sde_integrator = self.weighted_sde_integrator
        samples, _, _, _, _, acceptance_rate_list = weighted_sde_integrator.integrate_sde(
            x1=prior_samples.clone(),
            energy_function=energy_function,
            inverse_temperature=inverse_temp,
            annealing_factor_schedule=annealing_factor_schedule,
            resampling_interval=-1,
        )
        if not return_full_trajectory:
            samples = samples[-1]
        return samples

    def generate_samples(
        self,
        energy_function: BaseEnergyFunction,
        num_samples: int,
        inverse_temp: Optional[float] = 1.0,
        annealing_factor: Optional[float] = 1.0,
        annealing_factor_score: Optional[float] = None,
        return_logweights: Optional[bool] = False,
        weighted_sde_integrator: Optional[WeightedSDEIntegrator] = None,
    ) -> torch.Tensor:
        annealing_factor_schedule = self.hparams.annealing_factor_schedule(
            annealing_factor=annealing_factor
        )
        prior = self.hparams.partial_prior(
            scale=(
                self.hparams.noise_schedule.h(self.t_start)
                / annealing_factor_schedule.gamma(self.t_start)
            )
            ** 0.5,
            device=self.device,
        )

        prior_samples = prior.sample(num_samples)
        if annealing_factor_score is None:
            annealing_factor_score = annealing_factor

        if weighted_sde_integrator is None:
            weighted_sde_integrator = self.weighted_sde_integrator

        (
            samples,
            _,
            num_unique_idxs,
            sde_terms,
            acceptance_rate_list,
        ) = weighted_sde_integrator.integrate_sde(
            x1=prior_samples.clone(),
            energy_function=energy_function,
            inverse_temperature=inverse_temp,
            annealing_factor_schedule=annealing_factor_schedule,
            annealing_factor_score=annealing_factor_score,
        )
        if return_logweights:
            # reintegrate without resampling to get logweights, don't need as many samples
            samples_not_resampled, logweights, _, _, _ = weighted_sde_integrator.integrate_sde(
                x1=prior_samples.clone()[: self.hparams.inference_batch_size],
                energy_function=energy_function,
                resampling_interval=self.hparams.num_integration_steps + 1,
                inverse_temperature=inverse_temp,
                annealing_factor_schedule=annealing_factor_schedule,
                annealing_factor_score=annealing_factor_score,
            )
            return (
                samples,
                samples_not_resampled,
                logweights,
                num_unique_idxs,
                sde_terms,
                acceptance_rate_list,
            )

        return samples, num_unique_idxs, sde_terms, acceptance_rate_list

    def logsigma_stratified_loss(self, batch_t, batch_loss, num_bins=5, loss_name=None):
        """Stratify loss by binning t."""
        flat_losses = batch_loss.flatten().detach().cpu().numpy()
        flat_t = batch_t.flatten().detach().cpu().numpy()
        bin_edges = self.hparams.noise_schedule.get_ln_sigmat_bins(num_bins)
        # bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0)
        # import pdb; pdb.set_trace()
        bin_idx = np.digitize(flat_t, bin_edges)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)

        stratified_losses = {}
        if loss_name is None:
            loss_name = "loss"
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"{loss_name} ln_sigma=[{bin_start:.2f},{bin_end:.2f})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            stratified_losses[t_range] = range_loss
        return stratified_losses

    def get_loss(
        self,
        ht: torch.Tensor,
        x0: torch.Tensor,
        x0_energies: torch.Tensor,
        x0_forces: torch.Tensor,
        inverse_temp: float,
        energy_function: BaseEnergyFunction,
    ) -> torch.Tensor:
        h0 = self.hparams.noise_schedule.h(torch.zeros_like(ht))
        x0.requires_grad = True
        z = torch.randn_like(x0)
        z = self.maybe_remove_mean(z)
        x0 = self.maybe_remove_mean(x0)
        xt = x0 + z * ht[:, None] ** 0.5
        lambda_t = (ht + 1) / ht
        predicted_x0_scorenet = self.score_net.denoiser(ht, xt, inverse_temp, return_score=False)
        score_loss = self.get_score_loss(
            ht=ht,
            x0=x0,
            predicted_x0_scorenet=predicted_x0_scorenet,
            weights=lambda_t,
        )
        if self.hparams.get("only_train_score", False):
            zeros = torch.zeros_like(score_loss)
            return zeros, score_loss, zeros, zeros, zeros
        target_score_loss = self.get_target_score_loss(
            ht=ht,
            x0=x0,
            xt=xt,
            energy_function=energy_function,
            predicted_x0=predicted_x0_scorenet,
            true_force=x0_forces,
            weights=None,  # TODO: should we use lambda_t here?
            clipper=self.hparams.clipper,
        )
        energy_score_loss, predicted_dUt_dt, predicted_Ut = self.get_energy_score_loss(
            ht=ht,
            xt=xt,
            inverse_temp=inverse_temp,
            predicted_x0_scorenet=predicted_x0_scorenet,
            weights=lambda_t,
        )
        energy_matching_loss = self.get_energy_matching_loss(
            h0=h0,
            x0=x0,
            x0_energies=x0_energies,
            inverse_temp=inverse_temp,
            energy_function=energy_function,
        )
        dem_energy_loss = self.get_dem_energy_loss(
            ht=ht,
            xt=xt,
            energy_function=energy_function,
            predicted_Ut=predicted_Ut,
        )
        # L2 regularization on dU/dt
        dU_dt_regularization_loss = torch.sum(predicted_dUt_dt**2, dim=(-1))
        if self.hparams.loss_weights["dU_dt_regularization"] == 0:
            dU_dt_regularization_loss = torch.zeros_like(dU_dt_regularization_loss)
        return (
            energy_score_loss,
            score_loss,
            target_score_loss,
            dem_energy_loss,
            energy_matching_loss,
            dU_dt_regularization_loss,
        )

    def get_score_loss(
        self,
        ht: torch.Tensor,
        x0: torch.Tensor,
        predicted_x0_scorenet: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.hparams.loss_time_threshold["score"] > 0:
            assert (
                self.hparams.loss_weights["target_score"] > 0
            ), "target_score loss weight must be > 0 if score loss time threshold is > 0"
        if self.hparams.loss_weights["score"] == 0:
            return torch.zeros(x0.shape[0], device=x0.device)

        h_threshold = self.hparams.noise_schedule.h(self.hparams.loss_time_threshold["score"])
        time_mask = ht >= h_threshold
        if not time_mask.any():
            return torch.zeros_like(predicted_x0_scorenet)

        score_loss = torch.sum((predicted_x0_scorenet - x0) ** 2, dim=(-1))
        score_loss[~time_mask] = 0.0
        if weights is not None:
            score_loss = weights * score_loss
        return score_loss

    def get_energy_score_loss(
        self,
        ht: torch.Tensor,
        xt: torch.Tensor,
        inverse_temp: float,
        predicted_x0_scorenet: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ht.requires_grad = True
        t = self.hparams.noise_schedule.t(ht)

        if self.hparams.loss_weights["energy_score"] == 0:
            predicted_Ut = torch.zeros(xt.shape[0], device=xt.device)
            if not (self.hparams.loss_weights["dem_energy"] == 0):
                predicted_Ut = self.energy_net_forward_energy(ht, xt, inverse_temp)
            return (
                torch.zeros(xt.shape[0], device=xt.device),
                torch.zeros_like(predicted_Ut),
                predicted_Ut,
            )

        (
            predicted_x0_energynet,
            predicted_dUt_dht,
            predicted_Ut,
        ) = self.energy_net_denoiser_and_energy(
            ht=ht,
            xt=xt,
            beta=inverse_temp,
        )
        # chain rule
        predicted_dUt_dt = predicted_dUt_dht * self.hparams.noise_schedule.dh_dt(t)
        energy_score_loss = torch.sum(
            (predicted_x0_energynet - predicted_x0_scorenet.detach()) ** 2, dim=(-1)
        )
        if weights is not None:
            energy_score_loss = weights * energy_score_loss

        return energy_score_loss, predicted_dUt_dt, predicted_Ut

    def get_target_score_loss(
        self,
        ht: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        energy_function: BaseEnergyFunction,
        predicted_x0: torch.Tensor,
        true_force: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        clipper: Optional[Clipper] = None,
    ) -> torch.Tensor:
        if self.hparams.loss_weights["target_score"] == 0:
            return torch.zeros(predicted_x0.shape[0], device=x0.device)

        h_threshold = self.hparams.noise_schedule.h(
            self.hparams.loss_time_threshold["target_score"]
        )
        time_mask = ht < h_threshold
        if not time_mask.any():
            return torch.zeros(predicted_x0.shape[0], device=x0.device)
        x0 = x0[time_mask]
        ht = ht[time_mask]
        xt = xt[time_mask]
        predicted_x0 = predicted_x0[time_mask]

        if true_force is None:
            energy = -energy_function(x0).sum()
            nabla_U0 = torch.autograd.grad(energy, x0, create_graph=True)[0]  # -score
        else:
            nabla_U0 = -true_force[time_mask]
        if clipper is not None:
            nabla_U0 = clipper.clip_scores(nabla_U0)
        x0 = xt - nabla_U0 * ht[:, None]
        target_score_loss = torch.sum((x0 - predicted_x0) ** 2, dim=(-1))

        if weights is not None:
            weights = weights[time_mask]
            return weights * target_score_loss
        return target_score_loss

    def get_dem_energy_loss(
        self,
        ht: torch.Tensor,
        xt: torch.Tensor,
        energy_function: BaseEnergyFunction,
        predicted_Ut: torch.Tensor,
        energy_threshold: float = 1e3,
        time_threshold: float = 0.2,
    ) -> torch.Tensor:
        if self.hparams.loss_weights["dem_energy"] == 0:
            return torch.zeros_like(predicted_Ut)

        h_threshold = self.hparams.noise_schedule.h(torch.tensor(time_threshold))
        time_mask = ht < h_threshold
        ht = ht[time_mask]
        xt = xt[time_mask]
        predicted_Ut = predicted_Ut[time_mask]
        Ut_estimate = -estimate_Rt(
            ht=ht,
            x=xt,
            energy_function=energy_function,
            num_mc_samples=self.hparams.dem.num_mc_samples,
        )
        mask = Ut_estimate > energy_threshold
        loss = (Ut_estimate - predicted_Ut) ** 2
        loss = ~mask * loss
        return loss

    def get_dem_loss(
        self,
        ht: torch.Tensor,
        xt: torch.Tensor,
        energy_function: BaseEnergyFunction,
        predicted_nabla_Ut: torch.Tensor,
    ) -> torch.Tensor:
        nabla_Ut_estimate = -estimate_grad_Rt(
            ht=ht,
            x=xt,
            energy_function=energy_function,
            num_mc_samples=self.hparams.dem.num_mc_samples,
        )
        nabla_Ut_estimate = self.hparams.dem.clipper.clip_scores(nabla_Ut_estimate)
        return torch.sum((nabla_Ut_estimate - predicted_nabla_Ut) ** 2, dim=(-1))

    def get_energy_matching_loss(
        self,
        h0: torch.Tensor,
        x0: torch.Tensor,
        inverse_temp: float,
        energy_function: BaseEnergyFunction,
        x0_energies: Optional[torch.Tensor] = None,
        energy_threshold: float = 1e3,
    ) -> torch.Tensor:
        if self.hparams.loss_weights["energy_matching"] == 0:
            return torch.zeros(x0.shape[0], device=x0.device)

        if x0_energies is None:
            x0_energies = energy_function(x0)

        U0_true = -x0_energies
        mask = U0_true > energy_threshold

        # z = torch.randn_like(x0)
        # z = self.maybe_remove_mean(z)
        # x0 = x0 + z * h0[:, None] ** 0.5 # TODO: is this better?
        U0_pred = self.energy_net.forward_energy(h0, x0, inverse_temp)
        energy_matching_loss = (U0_true - U0_pred) ** 2
        energy_matching_loss = ~mask * energy_matching_loss
        return energy_matching_loss

    def dem_pre_training_step(self, x0_samples, prefix):
        ln_sigmat = self.hparams.dem.noise_schedule.sample_ln_sigma(
            len(x0_samples), device=x0_samples.device
        )
        ht = torch.exp(2 * ln_sigmat)
        inverse_temp = self.inverse_temperatures[0]

        xt = x0_samples + torch.randn_like(x0_samples) * ht[:, None] ** 0.5

        predicted_nabla_Ut = self.score_net(ht, xt, inverse_temp)

        with torch.enable_grad():
            dem_score_loss = self.get_dem_loss(
                ht=ht,
                xt=xt,
                energy_function=self.energy_functions[0],
                predicted_nabla_Ut=predicted_nabla_Ut,
            )
        dem_score_loss = dem_score_loss.mean()
        loss_dict = {
            f"{prefix}/dem_score_loss": dem_score_loss,
        }
        self.log_dict(loss_dict, sync_dist=True)
        return dem_score_loss

    def pre_training_step(self, x0_samples, x0_forces, prefix):
        ln_sigmat = self.hparams.dem.noise_schedule.sample_ln_sigma(
            len(x0_samples), device=x0_samples.device
        )
        ht = torch.exp(2 * ln_sigmat)
        inverse_temp = self.inverse_temperatures[0]

        xt = x0_samples + torch.randn_like(x0_samples) * ht[:, None] ** 0.5

        predicted_x0_scorenet = self.score_net.denoiser(ht, xt, inverse_temp, return_score=False)

        with torch.enable_grad():
            target_score_loss = self.get_target_score_loss(
                ht=ht,
                x0=x0_samples,
                xt=xt,
                energy_function=self.energy_functions[0],
                predicted_x0=predicted_x0_scorenet,
                true_force=x0_forces,
                clipper=self.hparams.dem.clipper,
            )
        target_score_loss = target_score_loss.mean()
        loss_dict = {
            f"{prefix}/pretrain_target_score_loss": target_score_loss,
        }
        self.log_dict(loss_dict, sync_dist=True)
        return target_score_loss

    def model_step(self, x0_samples, x0_energies, x0_forces, temp_index, prefix):
        ln_sigmat = self.hparams.noise_schedule.sample_ln_sigma(
            len(x0_samples), device=x0_samples.device
        )
        ht = torch.exp(2 * ln_sigmat)

        inverse_temp = self.inverse_temperatures[temp_index]

        with torch.enable_grad():
            (
                energy_score_loss,
                score_loss,
                target_score_loss,
                dem_energy_loss,
                energy_matching_loss,
                dU_dt_regularization_loss,
            ) = self.get_loss(
                ht,
                x0_samples,
                x0_energies,
                x0_forces,
                inverse_temp,
                self.energy_functions[temp_index],
            )

        should_log_stratified_energy_score = (
            self.hparams.loss_weights["energy_score"] != 0 and prefix == "train"
        )
        should_log_stratified_score = self.hparams.loss_weights["score"] != 0 and prefix == "train"
        if should_log_stratified_score:
            self.log_dict(
                self.logsigma_stratified_loss(
                    ln_sigmat, score_loss, loss_name="train/stratified/score_loss"
                ),
                sync_dist=True,
            )
        if should_log_stratified_energy_score:
            self.log_dict(
                self.logsigma_stratified_loss(
                    ln_sigmat,
                    energy_score_loss,
                    loss_name="train/stratified/energy_score_loss",
                ),
                sync_dist=True,
            )

        energy_score_loss = energy_score_loss.mean()
        score_loss = score_loss.mean()
        target_score_loss = target_score_loss.mean()
        dem_energy_loss = dem_energy_loss.mean()
        energy_matching_loss = energy_matching_loss.mean()
        dU_dt_regularization_loss = dU_dt_regularization_loss.mean()

        loss_weights = self.hparams.loss_weights
        loss = (
            loss_weights["energy_score"] * energy_score_loss
            + loss_weights["score"] * score_loss
            + loss_weights["target_score"] * target_score_loss
            + loss_weights["dem_energy"] * dem_energy_loss
            + loss_weights["energy_matching"] * energy_matching_loss
            + loss_weights["dU_dt_regularization"] * dU_dt_regularization_loss
        )

        # update and log metrics
        loss_dict = {
            f"{prefix}/loss": loss,
            f"{prefix}/energy_score_loss": energy_score_loss,
            f"{prefix}/score_loss": score_loss,
            f"{prefix}/energy_matching_loss": energy_matching_loss,
            f"{prefix}/target_score_loss": target_score_loss,
            f"{prefix}/dem_energy_loss": dem_energy_loss,
            f"{prefix}/dU_dt_regularization_loss": dU_dt_regularization_loss,
        }
        self.log_dict(loss_dict, sync_dist=True, prog_bar=prefix == "train")
        return loss

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch < self.hparams.dem.num_training_epochs:
            x0_samples, _, x0_forces, _ = self.sample_buffers[0].sample(
                self.hparams.dem.training_batch_size
            )
            return self.pre_training_step(x0_samples, x0_forces, prefix="train")

        if self.hparams.train_on_all_temps:
            active_inverse_temperatures = self.inverse_temperatures[
                : self.active_inverse_temperature_index + 1
            ]
            temp_index = np.random.randint(0, len(active_inverse_temperatures))
        else:
            # TODO: random inverse temperatures for each element in the batch
            temp_index = self.active_inverse_temperature_index

        # print(f"Training on temperature {self.temperatures[temp_index]:0.3f}")
        # print(f"length of sample buffer {len(self.sample_buffers[temp_index])}")

        x0_samples, x0_energies, x0_forces, _ = self.sample_buffers[temp_index].sample(
            self.hparams.training_batch_size
        )
        if self.is_molecule:
            x0_samples, x0_forces = self.data_augmentation(x0_samples, x0_forces)
        if not self.should_mean_free:
            x0_samples = self.com_data_augmentation(x0_samples)
        loss = self.model_step(x0_samples, x0_energies, x0_forces, temp_index, prefix="train")
        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer.current_epoch >= self.hparams.dem.num_training_epochs:
            return
        if (
            self.trainer.current_epoch % self.hparams.dem.check_val_every_n_epochs == 0
            and self.trainer.global_step > 0
        ):
            self.eval_epoch_end_dem("val")

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        logger.debug(f"Eval step {prefix}")

        try:
            num_samples = min(self.hparams.num_eval_samples, self.hparams.training_batch_size)
            temp_index = self.active_inverse_temperature_index

            energy_function = self.energy_functions[temp_index]

            if prefix == "test":
                true_x0_samples = energy_function.sample_test_set(num_samples)
            elif prefix == "val":
                true_x0_samples = energy_function.sample_val_set(num_samples)

            true_x0_energies, true_x0_forces = energy_function(true_x0_samples, return_force=True)
            loss = self.model_step(
                true_x0_samples,
                true_x0_energies,
                true_x0_forces,
                temp_index,
                prefix=f"{prefix}/temp={self.temperatures[temp_index]:0.3f}",
            )

            self.log(f"{prefix}/loss", loss, sync_dist=True)

        except Exception as e:
            logger.error(f"Error in eval step {prefix}: {e}")
            raise e

    def eval_epoch_end_dem(self, prefix: str):
        logger.debug(f"Started DEM eval epoch end {prefix}")
        wandb_logger = get_wandb_logger(self.loggers)
        energy_function = self.energy_functions[0]
        samples, _, _ = self.generate_samples(
            energy_function=energy_function,
            num_samples=self.hparams.dem.num_samples_to_generate_per_epoch,
            weighted_sde_integrator=self.weighted_sde_integrator_dem,
        )
        samples = samples.detach()
        samples_energy = energy_function(samples)

        self.sample_buffers[0].add(
            samples,
            samples_energy,
        )
        prefix_plot = f"{prefix}/dem"
        if self.is_molecule:
            self._log_dist_w2(prefix=prefix_plot, temp_index=0, generated_samples=samples)
        self._log_energy_distances(
            prefix=prefix,
            temp_index=0,
            generated_samples=samples,
        )
        energy_function.log_on_epoch_end(
            samples,
            samples_energy,
            wandb_logger,
            prefix=prefix_plot,
        )
        self.log(f"{prefix_plot}/energy_mean", -samples_energy.mean(), sync_dist=True)
        logger.debug(f"Finished eval epoch end DEM {prefix}")

    def eval_epoch_end(self, prefix: str):
        if self.hparams.ema_decay > 0:
            if self.hparams.eval_ema:
                self.score_net.backup()
                self.score_net.copy_to_model()
                self.energy_net.backup()
                self.energy_net.copy_to_model()
                self.energy_net.eval()
                self.score_net.eval()
                self.evaluate(prefix)
                self.score_net.restore_to_model()
                self.energy_net.restore_to_model()
            if self.hparams.eval_non_ema:
                self.evaluate(prefix + "/non_ema")
        else:
            self.evaluate(prefix)

    def evaluate(self, prefix: str):
        if self.trainer.current_epoch < self.hparams.dem.num_training_epochs:
            return
        logger.debug(f"Started eval epoch end {prefix}")
        wandb_logger = get_wandb_logger(self.loggers)
        index = self.active_inverse_temperature_index
        index_lower = index
        num_samples = self.hparams.num_eval_samples
        anneal_to_lower_temp = (index < len(self.inverse_temperatures) - 1) and (
            self.trainer.current_epoch + 1
        ) == self.update_temp_epoch[index]
        if anneal_to_lower_temp:
            num_samples = self.hparams.num_temp_annealed_samples_to_generate
            index_lower += 1
            if self.hparams.get("skip_temp", False):
                index_lower += self.hparams.get("num_skip_temps", 1)

        active_inverse_temperatures = self.inverse_temperatures[index : index_lower + 1]
        logger.debug(
            f"Active inverse temperatures: {active_inverse_temperatures} during epoch {self.trainer.current_epoch}"
        )
        inverse_temp = self.inverse_temperatures[index]
        inverse_lower_temp = self.inverse_temperatures[index_lower]
        annealing_factor = inverse_lower_temp / inverse_temp
        annealing_factor_score = annealing_factor
        if anneal_to_lower_temp:
            annealing_factor_score = self.hparams.get("annealing_factor_score", annealing_factor)
        logger.debug(
            f"Annealing factor is {annealing_factor} and Annealing factor score is {annealing_factor_score}"
        )
        energy_function = self.energy_functions[index_lower]
        temp = self.temperatures[index]
        temp_lower = self.temperatures[index_lower]
        logger.debug(
            f"Generating {num_samples}"
            + f" samples for temperature {temp:0.3f} annealed to temperature {temp_lower:0.3f}"
        )
        (
            samples,
            samples_not_resampled,
            logweights,
            num_unique_idxs,
            sde_terms,
            acceptance_rate_list,
        ) = self.generate_samples(
            energy_function=energy_function,
            num_samples=num_samples,
            return_logweights=True,
            inverse_temp=inverse_temp,
            annealing_factor=annealing_factor,
            annealing_factor_score=annealing_factor_score,
        )
        samples_energy, samples_forces = energy_function(samples, return_force=True)
        if self.hparams.debug_generation:
            samples_not_resampled = self.debug_generate_samples(
                energy_function=energy_function,
                num_samples=num_samples,
                inverse_temp=inverse_temp,
                annealing_factor=annealing_factor,
                annealing_factor_score=annealing_factor_score,
            )
        samples_not_resampled_energy = energy_function(samples_not_resampled)
        if index_lower != index:
            # mask out samples with high energy
            mask = (samples_energy > self.hparams.energy_masking_threshold) | (
                samples_energy < -self.hparams.energy_masking_threshold
            )
            if sum(mask) == len(mask):
                logger.warning(
                    f"All samples have energy above threshold {self.hparams.energy_masking_threshold}"
                )
                print(
                    f"All samples have energy above threshold {self.hparams.energy_masking_threshold}"
                )
                mask = torch.zeros_like(mask, dtype=torch.bool)
            if not self.should_mean_free:
                samples = remove_mean(samples, self.n_particles, self.n_spatial_dim)
            # fill the buffers
            self.sample_buffers[index_lower].add(
                samples[~mask],
                samples_energy[~mask],
                samples_forces[~mask],
            )
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # append time to avoid overwriting
            path = f"{output_dir}/buffer_samples_temperature_{temp_lower:0.3f}.pt"
            torch.save(samples, path)
            torch.save(samples_energy, path.replace("buffer_samples", "buffer_energies"))
            torch.save(
                samples_not_resampled, path.replace("buffer_samples", "samples_not_resampled")
            )
            logger.info(f"Saving {len(samples)} samples to {path}")

            # reinitialize the networks:
            if not self.hparams.train_on_all_temps and self.hparams.get(
                "reinitialize_networks", False
            ):
                logger.info(f"Reinitializing networks for temperature {temp:0.3f}")
                h_theta = self.hparams.net().to(self.device)
                # logger.info(f"{self.score_net.model.blocks[0].attn_out.weight}")
                self.energy_net.reinitialize(h_theta)
                self.score_net.reinitialize(copy.deepcopy(h_theta))

        logger.debug(
            f"Buffer size for temperature {temp_lower:0.3f} is {len(self.sample_buffers[index_lower])} at epoch {self.trainer.current_epoch}"
        )
        print(
            f"Buffer size for temp {temp_lower:0.3f} is {len(self.sample_buffers[index_lower])} at epoch {self.trainer.current_epoch}"
        )
        # select a subset of the generated samples to log
        if self.is_molecule:
            self._log_dist_w2(prefix="val", temp_index=index_lower, generated_samples=samples)
        self._log_energy_distances(
            prefix="val",
            temp_index=index_lower,
            generated_samples=samples,
        )

        prefix_plot = f"val/{temp:0.0f} to {temp_lower:0.0f}"
        for term in fields(SDETerms):
            if (
                term.name == "drift_X"
                or term.name == "drift_A"
                or getattr(sde_terms[0], term.name) is None
            ):
                continue
            self._log_sde_term(sde_terms, term.name, prefix=prefix_plot)
        del sde_terms

        # log annealing factor:
        self._log_annealing_factor(
            annealing_factor=annealing_factor,
            prefix=prefix_plot,
        )

        metrics = energy_function.log_on_epoch_end(
            samples,
            samples_energy,
            wandb_logger,
            latest_samples_not_resampled=samples_not_resampled,
            prefix=prefix_plot,
        )
        if metrics is not None:
            self.log_dict(metrics)
        self.log(f"val/energy_mean", -samples_energy.mean(), sync_dist=True)
        self.log(
            f"val/energy_mean_no_resampling",
            -samples_not_resampled_energy.mean(),
            sync_dist=True,
        )
        self.log("val/lower_temp", temp_lower, sync_dist=True)
        if self.hparams.resampling_interval != -1:
            self._log_logweights(
                logweights,
                prefix=prefix_plot,
            )
            self._log_std_logweights(
                logweights,
                prefix=prefix_plot,
            )
            self._log_num_unique_idxs(
                num_unique_idxs,
                prefix=prefix_plot,
            )
            self._log_acceptance_rate(
                acceptance_rate_list,
                prefix=prefix_plot,
            )

            if wandb_logger is not None:
                self.logger.experiment.log(
                    {
                        "1D Array Plot": wandb.plot.line_series(
                            xs=torch.linspace(1, 0, len(num_unique_idxs)).tolist(),
                            ys=[num_unique_idxs],
                            keys=["Number of Unique Indices"],
                            title=rf"val/temp= {temp:0.3f}, annealed to {temp_lower:0.3f}",
                            xname="Time",
                        )
                    }
                )
        logger.debug(f"Finished eval epoch end {prefix}")

    def on_test_epoch_end(self) -> None:
        if self.hparams.get("temps_to_anneal_test", False):
            logger.info("Found temps to anneal test")
            highest_temp = self.temperatures[0]
            inverse_temps_to_anneal = [
                (
                    torch.round(highest_temp / a, decimals=2),
                    torch.round(highest_temp / b, decimals=2),
                )
                for a, b in self.hparams.temps_to_anneal_test
            ]
        else:
            logger.info("No temps to anneal test")
            inverse_temps_to_anneal = [
                (self.inverse_temperatures[i], self.inverse_temperatures[i + 1])
                for i in range(len(self.inverse_temperatures) - 1)
            ]

        for temps in inverse_temps_to_anneal:
            inverse_temp = temps[0].item()
            inverse_lower_temp = temps[1].item()
            # get the index of the inverse temperature
            index = torch.nonzero(self.inverse_temperatures == inverse_temp)[0].item()
            index_lower = torch.nonzero(self.inverse_temperatures == inverse_lower_temp)[0].item()
            annealing_factor = inverse_lower_temp / inverse_temp
            logger.info(
                f"Generating {self.hparams.num_samples_to_save} samples for temperature {inverse_temp:0.3f} annealed to temperature {inverse_lower_temp:0.3f}"
            )
            logger.info(f"temp_index is {index} and temp_index_lower is {index_lower}")
            logger.info(f"Resampling interval is {self.hparams.resampling_interval}")
            final_samples, _, sde_terms, _ = self.generate_samples(
                energy_function=self.energy_functions[index_lower],
                num_samples=self.hparams.num_samples_to_save,
                inverse_temp=inverse_temp,
                annealing_factor=annealing_factor,
            )
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # append time to avoid overwriting
            path = f"{output_dir}/samples_temperature_{self.temperatures[index]:0.3f}_annealed_to_{self.temperatures[index_lower]}.pt"
            torch.save(final_samples, path)
            logger.info(f"Saving samples to {path}")

            # compute metrics on a subset of the generated samples
            batch_generated_samples = sample_from_tensor(
                final_samples, self.hparams.test_batch_size
            )
            self._log_energy_distances(
                inverse_temp,
                prefix="test",
                generated_samples=batch_generated_samples,
            )
            self._log_dist_w2(
                inverse_temp,
                prefix="test",
                generated_samples=batch_generated_samples,
            )
            prefix_plot = (
                f"test/inverse_temp= {inverse_temp:0.3f} annealed to {inverse_lower_temp:0.3f}"
            )
            for term in fields(SDETerms):
                if (
                    term.name == "drift_X"
                    or term.name == "drift_A"
                    or getattr(sde_terms[0], term.name) is None
                ):
                    continue
                self._log_sde_term(sde_terms, term.name, prefix=prefix_plot)
            del sde_terms

            energy_function = self.energy_functions[index + 1]
            samples_energy = energy_function(final_samples)
            wandb_logger = get_wandb_logger(self.loggers)
            energy_function.log_on_epoch_end(
                final_samples,
                samples_energy,
                wandb_logger,
                prefix=prefix_plot,
            )
            self.log(f"{prefix_plot}/energy_mean", -samples_energy.mean(), sync_dist=True)
            logger.debug("Finished eval epoch end test")

    def _log_annealing_factor(self, annealing_factor, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        annealing_factor_schedule = self.hparams.annealing_factor_schedule(
            annealing_factor=annealing_factor.cpu()
        )
        integration_times = torch.linspace(1, 0, self.hparams.num_integration_steps)
        annealing_factor_over_time = annealing_factor_schedule.gamma(integration_times)
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        axs.plot(integration_times, annealing_factor_over_time)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/annealing_factor", [img])
        plt.close(fig)

    def _log_logweights(self, logweights, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        # sample a few at random from the logweights
        idx = torch.randint(0, logweights.shape[1], (15,))
        logweights = logweights[:, idx].cpu().numpy()
        integration_times = torch.linspace(1, 0, logweights.shape[0])
        axs.plot(integration_times, logweights)

        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/annealing_logweights", [img])
        plt.close(fig)

    def _log_std_logweights(self, logweights, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return

        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        std_logweights = logweights.std(dim=1).cpu().numpy()
        integration_times = torch.linspace(1, 0, std_logweights.shape[0])
        axs.plot(integration_times, std_logweights)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/std_logweights", [img])
        plt.close(fig)

    def _log_sde_term(self, sde_terms, term, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return

        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        div_st_mean = torch.stack(
            [getattr(sde_terms[i], term).mean() for i in range(len(sde_terms))]
        )
        div_st_std = torch.stack(
            [getattr(sde_terms[i], term).std() for i in range(len(sde_terms))]
        )
        div_st_mean = div_st_mean.cpu().numpy()
        div_st_std = div_st_std.cpu().numpy()

        integration_times = torch.linspace(1, 0, div_st_mean.shape[0])
        axs.plot(integration_times, div_st_std, label="mean")
        # axs.fill_between(integration_times, div_st_mean - div_st_std, div_st_mean + div_st_std, alpha=0.3)

        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/{term}", [img])
        plt.close(fig)

    def _log_energy_distances(self, temp_index, generated_samples, prefix="val"):
        energy_function = self.energy_functions[temp_index]
        generated_energies = energy_function(generated_samples)
        if "test" in prefix:
            data_set = energy_function.sample_test_set(generated_samples.shape[0])
        else:
            data_set = energy_function.sample_val_set(generated_samples.shape[0])
        energies = energy_function(data_set)
        energy_distances_dict = energy_distances(
            generated_energies,
            energies,
            prefix=prefix,
            energy_threshold=self.hparams.energy_masking_threshold,
        )
        self.log_dict(
            energy_distances_dict,
            sync_dist=True,
        )

    def _log_dist_w2(self, temp_index, generated_samples, prefix="val"):
        energy_function = self.energy_functions[temp_index]

        if "test" in prefix:
            data_set = energy_function.sample_test_set(generated_samples.shape[0])
        else:
            data_set = energy_function.sample_val_set(generated_samples.shape[0])

        dist_w2 = (
            pot.emd2_1d(
                energy_function.interatomic_dist(generated_samples).cpu().numpy().reshape(-1),
                energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
            )
            ** 0.5
        )
        self.log(f"{prefix}/dist_w2", dist_w2, sync_dist=True)

    def _log_num_unique_idxs(self, num_unique_idxs, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        integration_times = torch.linspace(1, 0, len(num_unique_idxs))
        axs.plot(integration_times, num_unique_idxs)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/num_unique_idxs", [img])

    def _log_acceptance_rate(self, acceptance_rate_list, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None or acceptance_rate_list is None or len(acceptance_rate_list) == 0:
            return
        acceptance_rate_arr = np.array(acceptance_rate_list)
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        axs.plot(np.arange(len(acceptance_rate_arr)), acceptance_rate_arr, label="Acceptance Rate")
        axs.set_xlabel("Step")
        axs.set_ylabel("Acceptance Rate")
        axs.set_title("Acceptance Rate per Step")
        fig.canvas.draw()
        img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        wandb_logger.log_image(f"{prefix}/acceptance_rate", [img])
        # Log mean and first step acceptance rate as scalars
        self.log(
            f"{prefix}/acceptance_rate_mean", float(np.mean(acceptance_rate_arr)), sync_dist=True
        )
        self.log(f"{prefix}/acceptance_rate_first", float(acceptance_rate_arr[0]), sync_dist=True)

    def maybe_remove_mean(self, x):
        if self.should_mean_free:
            x = remove_mean(x, self.n_particles, self.n_spatial_dim)
        return x

    def populate_initial_buffer(self):
        logger.debug(f"Buffer 0 size: {len(self.sample_buffers[0])}")
        if len(self.sample_buffers[0]) > 0:
            logger.debug("Buffer 0 is already populated skipping populate")
            return
        # Initialize buffer 0 with train data
        if self.hparams.init_from_prior or self.hparams.dem.num_training_epochs > 0:
            prior = self.hparams.partial_prior(
                scale=self.hparams.noise_schedule.h(self.t_start) ** 0.5, device=self.device
            )
            init_states = prior.sample(self.hparams.num_init_samples)
        elif self.hparams.debug_init_buffer_path is not None:
            init_states = torch.load(self.hparams.debug_init_buffer_path, map_location=self.device)
            logger.debug(f"Loaded buffer 0 from {self.hparams.debug_init_buffer_path}")
        else:
            init_states = self.energy_functions[0].sample_train_set(self.hparams.num_init_samples)
        init_energies, init_forces = self.energy_functions[0](init_states, return_force=True)
        self.sample_buffers[0].add(init_states, init_energies, init_forces)
        logger.debug(f"Populated buffer 0 with {len(self.sample_buffers[0])} samples")

    def setup(self, stage: str) -> None:
        # lightning.seed_everything(self.hparams.seed + self.trainer.global_rank, workers=True)
        logger.debug(f"Setting up {stage}")
        self.energy_functions = {}
        if self.hparams.num_epochs_per_temp is not None:
            assert len(self.hparams.num_epochs_per_temp) == len(self.inverse_temperatures) - 1
            self.update_temp_epoch = (
                np.cumsum(self.hparams.num_epochs_per_temp) + self.hparams.dem.num_training_epochs
            )
            assert (
                self.update_temp_epoch % self.trainer.check_val_every_n_epoch == 0
            ).all(), (
                "update_temp_epoch values must be divisible by the trainer.check_val_every_n_epoch"
            )
            logger.debug(
                f"Update temp epochs: {self.update_temp_epoch} for inverse temperatures {self.inverse_temperatures}"
            )
        else:
            assert (
                len(self.inverse_temperatures) == 1
            ), "Need to specify num_epochs_per_temp in the config file"

        for temp_index, _ in enumerate(self.inverse_temperatures):
            self.energy_functions[temp_index] = self.hparams.energy_function(
                device=self.device,
                temperature=self.temperatures[temp_index],
                device_index=str(self.trainer.local_rank),
            )

        self.sample_buffers.to(self.device)
        for i in range(len(self.sample_buffers)):
            self.sample_buffers[i].init_generator(
                seed=self.hparams.seed + self.trainer.global_rank
            )
        self.populate_initial_buffer()

        if self.is_molecule:
            self.n_particles = self.energy_functions[0].n_particles
            self.n_spatial_dim = self.energy_functions[0].n_spatial_dim
            self.data_augmentation = Random3DRotationTransform(
                self.n_particles, self.n_spatial_dim
            )
            self.com_data_augmentation = CenterOfMassTransform(
                self.n_particles, self.n_spatial_dim
            )
        logger.debug(f"Setup {stage} done")
        self.reverse_sde.trainer = self.trainer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.score_net, EMA):
            self.score_net.update_ema()
        if isinstance(self.energy_net, EMA):
            self.energy_net.update_ema()
