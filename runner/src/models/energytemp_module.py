import copy
import logging
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import PIL
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.energy_net import EnergyNet
from src.models.components.noise_schedules import BaseNoiseSchedule
from src.models.components.score_net import ScoreNet
from src.models.components.sde_integration import WeightedSDEIntegrator
from src.models.components.sdes import SDETerms, VEReverseSDE
from src.models.components.utils import sample_from_tensor
from src.utils.data_utils import remove_mean
from torchmetrics import MeanMetric
from .components.score_estimator import estimate_Rt, estimate_grad_Rt
from .components.clipper import Clipper
from .components.noise_schedules import BaseNoiseSchedule
from .components.prioritised_replay_buffer import PrioritisedReplayBuffer
from .components.sdes import VEReverseSDE

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


def get_wandb_logger(loggers):
    """Gets the wandb logger if it is the list of loggers otherwise returns None."""
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break
    return wandb_logger


class energyTempModule(BaseLightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        clipper: Clipper,
        noise_schedule: BaseNoiseSchedule,
        partial_buffer: PrioritisedReplayBuffer,
        num_samples_to_generate_per_epoch: int,
        training_batch_size: int,
        num_integration_steps: int,
        lr_scheduler_update_frequency: int,
        num_samples_to_save: int,
        num_init_samples: int,
        temperatures: List[float],
        num_eval_samples: int,
        scale_diffusion: bool,
        test_batch_size: int,
        inference_batch_size: int,
        start_resampling_step: int,
        end_resampling_step: int,
        resampling_interval: int,
        num_mc_samples: int,
        num_epochs_per_temp: List[int],
        num_negative_time_steps: int,
        P_mean: float,
        P_std: float,
        resample_at_end: bool,
        compile: bool,
        loss_weights: dict,
        partial_prior=None,
        init_from_prior=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        h_theta = self.hparams.net()
        self.energy_net = EnergyNet(score_net=copy.deepcopy(h_theta))
        self.score_net = ScoreNet(model=h_theta)
        self.clipper = clipper

        self.reverse_sde = VEReverseSDE(
            energy_net=self.energy_net,
            noise_schedule=self.hparams.noise_schedule,
            score_net=self.score_net,
            pin_energy=False,
            debias_inference=True,
        )
        if self.hparams.dem.num_training_epochs > 0:
            self.dem_reverse_sde = VEReverseSDE(
                noise_schedule=self.hparams.dem.noise_schedule,
                score_net=self.score_net,
                debias_inference=False
            )


        self.weighted_sde_integrator = WeightedSDEIntegrator(
            sde=self.reverse_sde,
            num_integration_steps=self.hparams.num_integration_steps,
            reverse_time=True,
            time_range=1.0,
            no_grad=True,
            diffusion_scale=self.hparams.diffusion_scale,
            resampling_interval=self.hparams.resampling_interval,
            num_negative_time_steps=self.hparams.num_negative_time_steps,
            start_resampling_step=self.hparams.start_resampling_step,
            end_resampling_step=self.hparams.end_resampling_step,
            resample_at_end=self.hparams.resample_at_end,
            batch_size=self.hparams.inference_batch_size,
            lightning_module=self,
        )

        self.val_energy_w2 = MeanMetric()
        self.val_energy_w1 = MeanMetric()
        self.val_dist_w2 = MeanMetric()
        self.val_num_unique_idxs = MeanMetric()

    def generate_samples(
        self,
        prior,
        energy_function: BaseEnergyFunction,
        num_samples: int,
        inverse_temp: Optional[float] = 1.0,
        annealing_factor: Optional[float] = 1.0,
        return_full_trajectory: bool = False,
        return_logweights=False,
    ) -> torch.Tensor:
        prior_samples = prior.sample(num_samples)

        samples, _, num_unique_idxs, sde_terms = (
            self.weighted_sde_integrator.integrate_sde(
                x1=prior_samples.clone(),
                energy_function=energy_function,
                inverse_temperature=inverse_temp,
                annealing_factor=annealing_factor,
            )
        )
        if not return_full_trajectory:
            samples = samples[-1]

        if return_logweights:
            # reintegrate without resampling to get logweights, don't need as many samples
            samples_not_resampled, logweights, _, _ = self.weighted_sde_integrator.integrate_sde(
                x1=prior_samples.clone()[: self.hparams.inference_batch_size],
                energy_function=energy_function,
                resampling_interval=self.hparams.num_integration_steps + 1,
                inverse_temperature=inverse_temp,
                annealing_factor=annealing_factor,
            )
            return samples, samples_not_resampled[-1], logweights, num_unique_idxs, sde_terms

        return samples, num_unique_idxs, sde_terms

    def logsigma_stratified_loss(self, batch_t, batch_loss, num_bins=5, loss_name=None):
        """Stratify loss by binning t."""
        flat_losses = batch_loss.flatten().detach().cpu().numpy()
        flat_t = batch_t.flatten().detach().cpu().numpy()
        bin_edges = np.linspace(
            self.hparams.P_mean - 2 * self.hparams.P_std,
            self.hparams.P_mean + 2 * self.hparams.P_std,
            num_bins + 1,
        )
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
        inverse_temp: float,
        energy_function: BaseEnergyFunction,
    ) -> torch.Tensor:
        
        h0 = self.hparams.noise_schedule.h(torch.zeros_like(ht))
        x0.requires_grad = True
        z = torch.randn_like(x0)
        xt = x0 + z * ht[:, None] ** 0.5
        # TODO: should probably do weighting
        lambda_t = 1  # (ht + 1) / ht

        x0 = self.maybe_remove_mean(x0)
        xt = self.maybe_remove_mean(xt)

        predicted_x0_scorenet = self.score_net.denoiser(
            ht, xt, inverse_temp, return_score=False
        )
        predicted_x0_energynet, predicted_Ut = self.energy_net.denoiser_and_energy(
            ht, xt, inverse_temp
        )

        energy_score_loss = torch.sum(
            (predicted_x0_energynet - predicted_x0_scorenet.detach()) ** 2, dim=(-1)
        )
        score_loss = torch.sum(
            (predicted_x0_scorenet - x0) ** 2, dim=(-1)
        )
        target_score_loss = self.get_target_score_loss(
            ht=ht,
            x0=x0,
            xt=xt,
            energy_function=energy_function,
            predicted_x0=predicted_x0_scorenet, 
        )
        dem_energy_loss = self.get_dem_energy_loss(
            ht=ht,
            xt=xt,
            energy_function=energy_function,
            predicted_Ut=predicted_Ut,
        )
        energy_matching_loss = self.get_energy_matching_loss(
            h0=h0,
            x0=x0,
            inverse_temp=inverse_temp,
            energy_function=energy_function,
        )
        energy_score_loss = lambda_t * energy_score_loss
        score_loss = lambda_t * score_loss
        target_score_loss = lambda_t * target_score_loss

        return energy_score_loss, score_loss, target_score_loss, dem_energy_loss, energy_matching_loss
    
    def get_target_score_loss(
        self, 
        ht: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        energy_function: BaseEnergyFunction,
        predicted_x0: torch.Tensor,
        time_threshold: float = 0.2,
    ) -> torch.Tensor:
        if self.hparams.loss_weights["target_score"] == 0:
            return torch.zeros(predicted_x0.shape[0], device=x0.device)
        
        h_threshold = self.hparams.noise_schedule.h(torch.tensor(time_threshold))
        time_mask = ht > h_threshold
        x0 = x0[time_mask]
        ht = ht[time_mask]
        xt = xt[time_mask]
        predicted_x0 = predicted_x0[time_mask]

        energy = -energy_function(x0).sum()
        score = torch.autograd.grad(energy, x0, create_graph=True)[0]
        score = self.hparams.clipper.clip_scores(score)
        x0 = xt - score * ht[:, None]
        target_score_loss = torch.sum(
            (x0 - predicted_x0) ** 2, dim=(-1)
        )
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
        time_mask = ht > h_threshold
        ht = ht[time_mask]
        xt = xt[time_mask]
        predicted_Ut = predicted_Ut[time_mask]
        Ut_estimate = - estimate_Rt(
            ht=ht,
            x=xt,
            energy_function=energy_function,
            num_mc_samples=self.hparams.num_mc_samples,
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

        nabla_Ut_estimate = - estimate_grad_Rt(
            ht=ht,
            x=xt,
            energy_function=energy_function,
            num_mc_samples=self.hparams.num_mc_samples,
        )
        nabla_Ut_estimate = self.clipper.clip_scores(nabla_Ut_estimate)
        loss = torch.sum(
            (nabla_Ut_estimate - predicted_nabla_Ut) ** 2, dim=(-1)
        )

        return loss

    def get_energy_matching_loss(
        self,
        h0: torch.Tensor,
        x0: torch.Tensor,
        inverse_temp: float,
        energy_function: BaseEnergyFunction,
        energy_threshold: float = 1e3,
    ) -> torch.Tensor:
        
        if self.hparams.loss_weights["energy_matching"] == 0:
            return torch.zeros(x0.shape[0], device=x0.device)

        U0_true = -energy_function(x0)
        mask = U0_true > energy_threshold
        U0_pred = self.energy_net.forward_energy(h0, x0, inverse_temp)
        energy_matching_loss = (U0_true - U0_pred) ** 2
        energy_matching_loss = ~mask * energy_matching_loss
        return energy_matching_loss
    


    def pre_training_step(self, x0_samples, prefix):
        ln_sigmat = (
            torch.randn(len(x0_samples)).to(x0_samples.device) * self.hparams.P_std
            + self.hparams.P_mean
        )
        ht = torch.exp(2 * ln_sigmat)
        inverse_temp = self.inverse_temperatures[0]
        import ipdb; ipdb.set_trace()

        xt = x0_samples + torch.randn_like(x0_samples) * ht[:, None] ** 0.5

        predicted_nabla_Ut = self.score_net(ht, xt, inverse_temp)
        
        with torch.enable_grad():
            dem_score_loss = self.get_dem_loss(
                ht = ht,
                xt = xt,
                energy_function = self.energy_functions[0],
                predicted_nabla_Ut = predicted_nabla_Ut,
            )
        loss = dem_score_loss.mean()
        loss_dict = {
            f"{prefix}/dem_score_loss": dem_score_loss,
        }
        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )
        return loss


    def model_step(self, x0_samples, temp_index, prefix):
        ln_sigmat = (
            torch.randn(len(x0_samples)).to(x0_samples.device) * self.hparams.P_std
            + self.hparams.P_mean
        )
        ht = torch.exp(2 * ln_sigmat)

        inverse_temp = self.inverse_temperatures[temp_index]

        with torch.enable_grad():
            energy_score_loss, score_loss, target_score_loss, dem_energy_loss, energy_matching_loss = self.get_loss(
                ht, x0_samples, inverse_temp, self.energy_functions[temp_index]
            )

        if prefix == "train":
            self.log_dict(
                self.logsigma_stratified_loss(
                    ln_sigmat, score_loss, loss_name="train/stratified/score_loss"
                ),
                sync_dist=True,
            )
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

        loss_weights = self.hparams.loss_weights
        loss = (
            loss_weights["energy_score"] * energy_score_loss
            + loss_weights["score"] * score_loss
            + loss_weights["target_score"] * target_score_loss
            + loss_weights["dem_energy"] * dem_energy_loss
            + loss_weights["energy_matching"] * energy_matching_loss
        )

        # update and log metrics
        loss_dict = {
            f"{prefix}/loss": loss,
            f"{prefix}/energy_score_loss": energy_score_loss,
            f"{prefix}/score_loss": score_loss,
            f"{prefix}/target_score_loss": target_score_loss,
            f"{prefix}/dem_energy_loss": dem_energy_loss,
            f"{prefix}/energy_matching_loss": energy_matching_loss,
        }

        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
        )

        return loss

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch < self.hparams.dem.num_training_epochs:
            x0_samples, _, _ = self.buffers[0].sample(
                self.hparams.dem.training_batch_size
            )   
            return self.pre_training_step(x0_samples, prefix="train")
        
        active_inverse_temperatures = self.inverse_temperatures[:self.active_inverse_temperature_index+1]
        # TODO: random inverse temperatures for each element in the batch
        temp_index = np.random.randint(0, len(active_inverse_temperatures))
        x0_samples, _, _ = self.buffers[temp_index].sample(
            self.hparams.training_batch_size
        )

        loss = self.model_step(x0_samples, temp_index, prefix="train")

        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer.current_epoch % self.hparams.dem.check_val_every_n_epochs == 0:
                self.eval_epoch_end_dem("val")

    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        """Perform a single eval step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        logger.debug(f"Eval step {prefix}")
        val_loss = 0.0
        for temp_index, inverse_temp in enumerate(self.inverse_temperatures):
            energy_function = self.energy_functions[temp_index]

            if prefix == "test":
                true_x0_samples = energy_function.sample_test_set(
                    self.hparams.num_eval_samples
                )
            elif prefix == "val":
                true_x0_samples = energy_function.sample_val_set(
                    self.hparams.num_eval_samples
                )

            loss = self.model_step(
                true_x0_samples,
                temp_index,
                prefix=f"{prefix}/inv_temp={inverse_temp:0.3f}",
            )
            val_loss += loss

        self.log(f"{prefix}/loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)


    def eval_epoch_end_dem(self, prefix: str):
        logger.debug(f"Started DEM eval epoch end {prefix}")
        wandb_logger = get_wandb_logger(self.loggers)
        energy_function = self.energy_functions[0]
        samples, _, _ = self.generate_samples(
            prior=self.priors[0],
            energy_function=energy_function,
            num_samples=self.hparams.dem.num_samples_to_generate_per_epoch,
        )
        samples_energy = energy_function(samples)
                
        self.buffers[0].add(
            samples,
            samples_energy,
        )

        prefix_plot = f"{prefix}/dem"
        if self.is_molecule:
            self._log_dist_w2(
                prefix=prefix_plot, temp_index=0, generated_samples=samples
            )
        self._log_energy_w2(
            prefix=prefix_plot, temp_index=0, generated_samples=samples,
        )
        
        energy_function.log_on_epoch_end(
            samples,
            samples_energy,
            wandb_logger,
            prefix=prefix_plot,
        )
        self._log_energy_mean(
            -samples_energy,
            prefix=prefix_plot,
        )
        logger.debug(f"Finished eval epoch end DEM {prefix}")



    def eval_epoch_end(self, prefix: str):
        logger.debug(f"Started eval epoch end {prefix}")
        wandb_logger = get_wandb_logger(self.loggers)

        active_inverse_temperatures = [self.inverse_temperatures[self.active_inverse_temperature_index]]
        temp_index = self.active_inverse_temperature_index
        inverse_temp = self.inverse_temperatures[temp_index]

        if (self.trainer.current_epoch > 0 
            and (self.trainer.current_epoch+1) == self.update_temp_epoch[self.active_inverse_temperature_index] 
            and self.active_inverse_temperature_index< len(self.inverse_temperatures) - 1):
            # update active inverse temperatures
            active_inverse_temperatures = self.inverse_temperatures[self.active_inverse_temperature_index:self.active_inverse_temperature_index+2]
            temp_index_lower = self.active_inverse_temperature_index + 1
            self.active_inverse_temperature_index = temp_index_lower
            num_samples = self.hparams.num_samples_to_generate_per_epoch
        else:
            temp_index_lower = temp_index
            num_samples = self.hparams.num_eval_samples

        logger.debug(f"Active inverse temperatures: {active_inverse_temperatures} during epoch {self.trainer.current_epoch}")
        
        # for inverse_temp in active_inverse_temperatures:
        logger.debug(f"Started eval epoch end for inverse_temp {inverse_temp:0.3f}")

        inverse_lower_temp = self.inverse_temperatures[temp_index_lower]
        energy_function = self.energy_functions[temp_index_lower]

        logger.debug(f"inverse_temp is {inverse_temp:0.3f} and inverse_lower_temp is {inverse_lower_temp:0.3f}")
        logger.debug(f"temp_index is {temp_index} and temp_index_lower is {temp_index_lower}")

        logger.debug(f"Generating {self.hparams.num_samples_to_generate_per_epoch}" 
                        + f" samples for inverse_temp {inverse_temp:0.3f} annealed to {inverse_lower_temp:0.3f}")
        samples, samples_not_resampled, logweights, num_unique_idxs, sde_terms = self.generate_samples(
            prior=self.priors[temp_index_lower],
            energy_function=energy_function,
            num_samples=num_samples,
            return_logweights=True,
            inverse_temp=inverse_temp,
            annealing_factor=inverse_lower_temp / inverse_temp,
        )
        samples_energy = energy_function(samples)
        if temp_index_lower != temp_index:
            # fill the buffers
            self.buffers[temp_index_lower].add(
                samples, 
                samples_energy,
            )
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # append time to avoid overwriting
            path = f"{output_dir}/buffer_samples_temperature_{inverse_lower_temp:0.3f}.pt"
            torch.save(samples, path)
            torch.save(samples_energy, path.replace("buffer_samples", "buffer_energies"))
            logger.info(f"Saving samples to {path}")
        logger.debug(
            f"Buffer size for inverse_temp {inverse_lower_temp:0.3f} is {len(self.buffers[temp_index_lower])} at epoch {self.trainer.current_epoch}"
            )

        print(f"Buffer size for inverse_temp {inverse_lower_temp:0.3f} is {len(self.buffers[temp_index_lower])} at epoch {self.trainer.current_epoch}")
        # select a subset of the generated samples to log
        if self.is_molecule:
            self._log_dist_w2(
                prefix="val", temp_index=temp_index_lower, generated_samples=samples
            )
        self._log_energy_w2(
            prefix="val", temp_index=temp_index_lower, generated_samples=samples,
        )

        prefix_plot = f"val/inv_temp= {inverse_temp:0.3f} annealed to {inverse_lower_temp:0.3f}"
        for term in fields(SDETerms):
            if term.name == "drift_X" or term.name == "drift_A":
                continue
            self._log_sde_term(sde_terms, term.name, prefix=prefix_plot)

        energy_function.log_on_epoch_end(
            samples,
            samples_energy,
            wandb_logger,
            latest_samples_not_resampled=samples_not_resampled,
            prefix=prefix_plot,
        )
        self._log_energy_mean(
            -samples_energy,
            prefix=prefix_plot,
        )
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

            if wandb_logger is not None:
                self.logger.experiment.log(
                    {
                        "1D Array Plot": wandb.plot.line_series(
                            xs=torch.linspace(1, 0, len(num_unique_idxs)).tolist(),
                            ys=[num_unique_idxs],
                            keys=["Number of Unique Indices"],
                            title=rf"val / $\beta$= {inverse_temp:0.3f}, $\gamma$= {inverse_lower_temp:0.3f}",
                            xname="Time",
                        )
                    }
                )
        logger.debug(f"Finished eval epoch end {prefix}")

    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        for temp_index, inverse_temp in enumerate(self.inverse_temperatures[:-1]):
            inverse_lower_temp = self.inverse_temperatures[temp_index + 1]
            final_samples = []

            batch_size = self.hparams.num_eval_samples
            n_batches = self.hparams.num_samples_to_save // batch_size
            logger.info(
                f"Generating {n_batches} batches of annealed samples of size {batch_size}."
            )
            logger.info(f"Resampling interval is {self.hparams.resampling_interval}")

            for i in range(n_batches):
                start = time.time()
                samples, _, _, _ = self.generate_samples(
                    prior=self.priors[temp_index + 1],
                    energy_function=self.energy_functions[temp_index + 1],
                    num_samples=self.hparams.num_samples_to_save,
                    inverse_temp=inverse_temp,
                    annealing_factor=inverse_lower_temp / inverse_temp,
                )
                final_samples.append(samples)
                end = time.time()
                logger.info(f"batch {i} took {end - start:0.2f}s")

            final_samples = torch.cat(final_samples, dim=0)
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # append time to avoid overwriting
            path = f"{output_dir}/samples_temperature_{inverse_lower_temp}_{self.hparams.num_samples_to_save}.pt"
            torch.save(final_samples, path)
            logger.info(f"Saving samples to {path}")

            # compute metrics on a subset of the generated samples
            batch_generated_samples = sample_from_tensor(
                final_samples, self.hparams.test_batch_size
            )
            # log energy w2
            self._log_energy_w2(
                inverse_temp,
                prefix="test",
                test_generated_samples=batch_generated_samples,
            )
            self._log_dist_w2(
                inverse_temp,
                prefix="test",
                test_generated_samples=batch_generated_samples,
            )

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

        # limit yaxis
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb_logger.log_image(f"{prefix}/annealing_logweights", [img])

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
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb_logger.log_image(f"{prefix}/std_logweights", [img])

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
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb_logger.log_image(f"{prefix}/{term}", [img])

    def _log_energy_w2(self, temp_index, generated_samples, prefix="val"):
        energy_function = self.energy_functions[temp_index]
        generated_energies = energy_function(generated_samples)

        if "test" in prefix:
            data_set = energy_function.sample_test_set(self.hparams.num_eval_samples)
        else:
            data_set = energy_function.sample_val_set(self.hparams.num_eval_samples)

        energies = energy_function(energy_function.normalize(data_set))
        energy_w2 = pot.emd2_1d(
            energies.cpu().numpy(), generated_energies.cpu().numpy()
        )**0.5
        self.log(
            f"{prefix}/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _log_dist_w2(self, temp_index, generated_samples, prefix="val"):
        energy_function = self.energy_functions[temp_index]

        if "test" in prefix:
            data_set = energy_function.sample_test_set(self.test_batch_size)
        else:
            data_set = energy_function.sample_val_set(self.hparams.num_eval_samples)

        dist_w2 = pot.emd2_1d(
            energy_function.interatomic_dist(generated_samples)
            .cpu()
            .numpy()
            .reshape(-1),
            energy_function.interatomic_dist(data_set).cpu().numpy().reshape(-1),
        )**0.5
        self.log(
            f"{prefix}/dist_w2",
            self.val_dist_w2(dist_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
    
    def _log_num_unique_idxs(self, num_unique_idxs, prefix="val"):
        wandb_logger = get_wandb_logger(self.loggers)
        if wandb_logger is None:
            return
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        integration_times = torch.linspace(1, 0, len(num_unique_idxs))
        axs.plot(integration_times, num_unique_idxs)
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb_logger.log_image(f"{prefix}/num_unique_idxs", [img])

    def _log_energy_mean(
        self,
        samples_energy,
        prefix="val",
    ):
        self.log(
            f"{prefix}/energy_mean",
            samples_energy.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def maybe_remove_mean(self, x):
        if self.is_molecule:
            x = remove_mean(
                x, self.n_particles, self.n_spatial_dim
            )
        return x

    def setup(self, stage: str) -> None:
        self.energy_functions = {}
        self.priors = {}
        self.buffers = {}
        self.last_samples = {}
        self.last_energies = {}


        temperatures = torch.tensor(self.hparams.temperatures)

        logger.debug(f"Temperatures: {temperatures}")

        self.inverse_temperatures = torch.round(temperatures[0] / temperatures, decimals=2).to(
                self.device
            )
        self.temperatures = temperatures

        self.active_inverse_temperature_index = 0

        logger.debug(f"Inverse Temperatures: {self.inverse_temperatures}")

        times = torch.linspace(1, 0, self.hparams.num_integration_steps + 1)
        t_start = times[self.hparams.start_resampling_step]

        # import ipdb; ipdb.set_trace()
        if self.hparams.num_epochs_per_temp is not None:
            assert len(self.hparams.num_epochs_per_temp) == len(self.inverse_temperatures) - 1 
            self.update_temp_epoch = np.cumsum(self.hparams.num_epochs_per_temp) + self.hparams.dem.num_training_epochs
            assert (self.update_temp_epoch % self.trainer.check_val_every_n_epoch == 0).all(), \
                "update_temp_epoch values must be divisible by the trainer.check_val_every_n_epoch"
            logger.debug(f"Update temp epochs: {self.update_temp_epoch} for inverse temperatures {self.inverse_temperatures}")
        else: 
            assert len(self.inverse_temperatures) - 1  == 0, \
            "Need to specify num_epochs_per_temp in the config file"

        for temp_index, inverse_temp in enumerate(self.inverse_temperatures):
            self.energy_functions[temp_index] = self.hparams.energy_function(
                device=self.device,
                temperature=temperatures[temp_index],
                device_index=str(self.trainer.local_rank)
            )
            self.priors[temp_index] = self.hparams.partial_prior(
                device=self.device,
                scale=(self.hparams.noise_schedule.h(t_start) / inverse_temp) ** 0.5,
            )
            self.buffers[temp_index] = self.hparams.partial_buffer(device=self.device)
            self.last_samples[temp_index] = None
            self.last_energies[temp_index] = None

            if self.hparams.init_from_prior or self.hparams.dem.num_training_epochs > 0:
                init_states = self.priors[temp_index].sample(
                    self.hparams.num_init_samples
                )
            else:
                init_states = self.energy_functions[0].sample_test_set(
                    self.hparams.num_init_samples
                )
            init_energies = self.energy_functions[temp_index](init_states)

            if temp_index == 0:
                self.buffers[temp_index].add(init_states, init_energies)

        self.is_molecule = self.energy_functions[0].is_molecule
        if self.is_molecule:
            self.n_particles = self.energy_functions[0].n_particles
            self.n_spatial_dim = self.energy_functions[0].n_spatial_dim


if __name__ == "__main__":
    _ = energyTempModule(
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

    # def get_energy_loss(self,
    #                     ht: torch.Tensor,
    #                     x0: torch.Tensor,
    #                     inverse_temp: float) -> torch.Tensor:
    #     x0.requires_grad = True
    #     z = torch.randn_like(x0)
    #     xt = x0 + z * ht[:, None] ** 0.5

    #     if self.is_molecule:
    #         xt = remove_mean(
    #             xt,
    #             self.n_particles,
    #             self.n_spatial_dim,
    #         )

    #     predicted_score = - self.energy_net.forward(ht, xt, inverse_temp)

    #     epsilon = -z

    #     lambda_t = (ht + 1) / ht
    #     score_loss = torch.sum(
    #         (predicted_score * ht[:, None] ** 0.5 - epsilon) ** 2, dim=(-1)
    #     )
    #     score_loss = score_loss.mean()
    #     return score_loss
