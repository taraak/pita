from typing import Any, Optional
import torch
import wandb
import PIL
from .dem_module import *
from src.models.components.sdes import VEReverseSDE
from src.models.components.utils import sample_from_tensor
from src.models.components.temperature_schedules import BaseInverseTempSchedule


class tempDEMLitModule(DEMLitModule):
    def __init__(
        self,
        annealed_energy: BaseEnergyFunction,
        num_eval_samples: int,
        scale_diffusion: bool,
        temperature_schedule: BaseInverseTempSchedule,
        annealed_test_batch_size: int,
        start_resampling_step: int,
        annealed_clipper: Clipper,
        resampling_strategy: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
    def generate_samples(
        self,
        reverse_sde: VEReverseSDE,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale = None,
        resampling_interval=None,
        return_logweights=False,
        num_langevin_steps=1,
        batch_size = None,
        num_negative_time_steps=-1,
        prior_samples = None,
        logq_prior_samples = None,
        start_resampling_step=0,
        return_num_unique_idxs=False,
        resampling_strategy="systematic",
    ) -> torch.Tensor:
        
        diffusion_scale = diffusion_scale or self.hparams.diffusion_scale #TODO: should we use same diff scale?
        reverse_sde = reverse_sde or self.reverse_sde

        if num_negative_time_steps == -1:
            num_negative_time_steps = self.hparams.num_negative_time_steps

        if prior_samples is None:
            prior_samples = self.prior.sample(num_samples)
            logq_prior_samples = self.prior.log_prob(prior_samples)

        samples, _, num_unique_idxs = self.integrate(
            reverse_sde=reverse_sde,
            samples=prior_samples.clone(),
            logq_samples=logq_prior_samples.clone(),
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale, 
            resampling_interval=resampling_interval,
            num_langevin_steps=num_langevin_steps,
            batch_size=batch_size,
            num_negative_time_steps=num_negative_time_steps,
            start_resampling_step=start_resampling_step,
            resampling_strategy=resampling_strategy,
        )
        # TODO: When returning the weights for plotting, I am not doing additional langevin steps
        if return_logweights:
            # reintegrate without resampling to get logweights, don't need as many samples
            _, logweights, _ = self.integrate(
                reverse_sde=reverse_sde,
                samples=prior_samples.clone()[:batch_size],
                logq_samples=logq_prior_samples.clone()[:batch_size],
                reverse_time=True,
                return_full_trajectory=return_full_trajectory,
                diffusion_scale=diffusion_scale, 
                resampling_interval=self.num_integration_steps+1,
                num_langevin_steps=1,
                batch_size=batch_size,
                num_negative_time_steps=self.hparams.num_negative_time_steps, #TODO: Should we do any negative time here?
                start_resampling_step=start_resampling_step,
                resampling_strategy=resampling_strategy,
            )
            return samples, logweights, num_unique_idxs 
    
        if return_num_unique_idxs:
            return samples, num_unique_idxs
        return samples
    
    def integrate(
        self,
        reverse_sde: VEReverseSDE,
        samples: torch.Tensor,
        logq_samples: torch.Tensor,
        reverse_time: bool,
        diffusion_scale: float,
        resampling_interval: int,
        num_langevin_steps: int,
        num_negative_time_steps: int,
        start_resampling_step: int,
        resampling_strategy: str,
        batch_size=None,
        no_grad=True,
        return_full_trajectory=False,
    ) -> torch.Tensor:
        trajectory, logweights, num_unique_idxs = integrate_sde(
            reverse_sde,
            samples,
            logq_samples,
            self.num_integration_steps,
            self.energy_function,
            diffusion_scale=diffusion_scale,
            reverse_time=reverse_time,
            no_grad=no_grad,
            num_negative_time_steps=num_negative_time_steps,
            num_langevin_steps=num_langevin_steps,
            resampling_interval=resampling_interval,
            batch_size=batch_size,
            start_resampling_step=start_resampling_step,
            resampling_strategy=resampling_strategy,
        )
        if return_full_trajectory:
            trajectory, logweights, num_unique_idxs

        return trajectory[-1], logweights, num_unique_idxs
    
    def eval_step(self, prefix: str, batch: torch.Tensor, batch_idx: int) -> None:
        if self.trainer_called:
            super().eval_step(prefix, batch, batch_idx)
        return

    def eval_epoch_end(self, prefix: str):
        if self.trainer_called:
            super().eval_epoch_end(prefix)

        if self.eval_count % 5 == 0:
            wandb_logger = get_wandb_logger(self.loggers)

            prior_samples = self.annealed_prior.sample(self.hparams.num_eval_samples)
            logq_prior_samples = self.annealed_prior.log_prob(prior_samples)


            self.annealed_reverse_sde = VEReverseSDE(self.net, self.hparams.noise_schedule,
                                            temperature_schedule=self.temperature_schedule,
                                            scale_diffusion=self.hparams.scale_diffusion, 
                                            clipper=self.hparams.annealed_clipper
                                            )

            self.last_samples_annealed, logweights, num_unique_idxs = self.generate_samples(
                reverse_sde=self.annealed_reverse_sde,
                return_logweights=True,
                resampling_interval=self.hparams.resampling_interval,
                diffusion_scale=self.hparams.diffusion_scale,  #TODO: what should the diffusion scale be?
                num_langevin_steps=self.hparams.num_langevin_steps,
                batch_size=self.hparams.num_samples_to_generate_per_epoch,
                prior_samples=prior_samples,
                logq_prior_samples=logq_prior_samples,
                num_negative_time_steps=self.hparams.num_negative_time_steps, #TODO: Should we do any negative time here?
                resampling_strategy=self.hparams.resampling_strategy,
            )
            self.last_energies_annealed = self.annealed_energy(self.last_samples_annealed)

            print(f"Generated {self.last_samples_annealed.shape[0]} annealed samples at temperature {self.annealed_energy.temperature}.")

            self.annealed_energy.log_on_epoch_end(
                self.last_samples_annealed,
                self.last_energies_annealed,
                wandb_logger,
                prefix="temp_annealed_samples",
            )

            self._log_energy_mean(self.last_energies_annealed, prefix="val/temp_annealed")

            if self.hparams.resampling_interval!=-1:
                self._log_logweights(logweights, prefix="val")
                self._log_std_logweights(logweights, prefix="val")
                self._log_num_unique_idxs(num_unique_idxs, prefix="val")
                self.logger.experiment.log({
                    "1D Array Plot": wandb.plot.line_series(
                        xs=torch.linspace(1, 0, len(num_unique_idxs)).tolist(),  # X-axis: indices or time points
                        ys=[num_unique_idxs],                 # Y-axis: values
                        keys=["Value"],                       # Line legend
                        title="Number of Unique Indices",     # Plot title
                        xname="Time",                         # X-axis title
            )
        })
                
            
        self.eval_count +=1


    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)
        super().on_test_epoch_end()

        # Compute metrics for the annealed energy
        batch_annealed_samples = sample_from_tensor(self.last_samples_annealed,
                                                    self.hparams.annealed_test_batch_size)
        batch_annealed_energies = self.annealed_energy(batch_annealed_samples)
        batch_test_samples = self.annealed_energy.sample_test_set(self.hparams.annealed_test_batch_size)
        batch_test_energies = self.annealed_energy(batch_test_samples)

        names, dists = compute_distribution_distances(
            self.annealed_energy.unnormalize(batch_annealed_samples)[:, None],
            batch_test_samples[:, None], self.annealed_energy
        )
        names = [f"test/temp_annealed/{name}" for name in names]
        d = dict(zip(names, dists))
        self.log_dict(d, sync_dist=True)

        energy_w2 = pot.emd2_1d(
            batch_test_energies.cpu().numpy(),
            batch_annealed_energies.cpu().numpy()
        )
        self.log(
            "test/temp_annealed/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        energy_w1 = pot.emd2_1d(
            batch_test_energies.cpu().numpy(),
            batch_annealed_energies.cpu().numpy(),
            metric="euclidean"
        )
        self.log(
            "test/temp_annealed/energy_w1",
            self.val_energy_w1(energy_w1),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.annealed_energy.is_molecule:
            dist_w2 = pot.emd2_1d(
            self.annealed_energy.interatomic_dist(batch_annealed_samples).cpu().numpy().reshape(-1),
            self.annealed_energy.interatomic_dist(batch_test_samples).cpu().numpy().reshape(-1)
            )
            self.log(
                "test/temp_annealed/dist_w2",
                self.val_dist_w2(dist_w2),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.annealed_energy.is_molecule:
            self._log_dist_w2(prefix="test/temp_annealed", energy_function=self.annealed_energy)
            self._log_dist_total_var(prefix="test/temp_annealed", energy_function=self.annealed_energy)

        # Generate annealed samples to save
        final_samples = []

        if self.num_samples_to_save >  self.hparams.num_eval_samples:   
            batch_size = self.hparams.num_eval_samples
            n_batches = self.num_samples_to_save // batch_size
            print(f"Generating {n_batches} batches of annealed samples of size {batch_size}.")
            print(f"Resampling interval is {self.hparams.resampling_interval}")
            prior_samples = self.annealed_prior.sample(batch_size)
            logq_prior_samples = self.annealed_prior.log_prob(prior_samples)

            for i in range(n_batches):
                start = time.time()
                samples = self.generate_samples(
                        reverse_sde=self.annealed_reverse_sde,
                        return_logweights=False,
                        diffusion_scale=self.hparams.diffusion_scale,
                        resampling_interval=self.hparams.resampling_interval,
                        num_langevin_steps=self.hparams.num_langevin_steps,
                        batch_size=self.hparams.num_samples_to_generate_per_epoch,
                        prior_samples=prior_samples,
                        logq_prior_samples=logq_prior_samples,
                        num_negative_time_steps=self.hparams.num_negative_time_steps,
                        resampling_strategy=self.hparams.resampling_strategy,
                    )
                final_samples.append(samples
                )
                end = time.time()
                print(f"batch {i} took {end - start:0.2f}s")

                if i==0:
                    self.annealed_energy.log_on_epoch_end(
                        samples,
                        self.annealed_energy(samples),
                        wandb_logger,
                        prefix="test/temp_annealed_samples",
                    )

            final_samples = torch.cat(final_samples, dim=0)

        else:
            final_samples = sample_from_tensor(self.last_samples_annealed,
                                               self.hparams.num_samples_to_save)

        
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # append time to avoid overwriting
        path = f"{output_dir}/samples_temperature_{self.annealed_energy.temperature}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os
        os.makedirs(self.annealed_energy.name, exist_ok=True)
        path2 = f"{self.annealed_energy.name}/samples_temperature_{self.annealed_energy.temperature}_{self.hparams.version}_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path2)
        print(f"Saving samples to {path2}")


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

        #limit yaxis
        axs.set_xlabel("Integration time")
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        wandb_logger.log_image(
            f"{prefix}/annealing_logweights", [img]
        )

    def _log_std_logweights(self, logweights, prefix="val"):
        wandb_logger= get_wandb_logger(self.loggers)
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
        wandb_logger.log_image(
            f"{prefix}/std_logweights", [img]
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
        wandb_logger.log_image(
            f"{prefix}/num_unique_idxs", [img]
        )

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.annealed_energy = self.hparams.annealed_energy(device=self.device)

        inverse_temp = self.energy_function.temperature / self.annealed_energy.temperature
        self.temperature_schedule = self.hparams.temperature_schedule(inverse_temp)
        print("Inverse Temperature is", inverse_temp)

        times = torch.linspace(1, 0, self.num_integration_steps + 1)
        t_start = times[self.hparams.start_resampling_step]
        print(f"Resampling will start at time {t_start}")
        self.annealed_prior = self.partial_prior(
            device=self.device, scale=(self.noise_schedule.h(t_start) / inverse_temp) ** 0.5
        )
        
        # self.annealed_prior = self.partial_prior(
        #     device=self.device, scale=(self.noise_schedule.h(1) / inverse_temp) ** 0.5
        # )


        self.hparams.annealed_clipper.energy_function=self.energy_function

        self.eval_count = 0

        # self.annealed_reverse_sde = VEReverseSDE(self.net, self.hparams.noise_schedule,
        #                                 temperature_schedule=self.temperature_schedule,
        #                                 scale_diffusion=self.hparams.scale_diffusion, 
        #                                 clipper=self.hparams.annealed_clipper
        #                                 )


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