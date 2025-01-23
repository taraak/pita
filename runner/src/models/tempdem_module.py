from typing import Any, Optional
import torch
import PIL
from .dem_module import *
from src.models.components.sdes import VEReverseSDE
from src.models.components.utils import sample_from_tensor


class tempDEMLitModule(DEMLitModule):
    def __init__(
        self,
        annealed_energy: BaseEnergyFunction,
        num_eval_samples: int,
        scale_diffusion: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
    def generate_samples(
        self,
        reverse_sde: VEReverseSDE = None,
        num_samples: Optional[int] = None,
        return_full_trajectory: bool = False,
        diffusion_scale = None,
        resampling_interval=None,
        return_logweights=False,
        num_langevin_steps=1,
        batch_size=None,
        num_negative_time_steps=-1,
        prior_samples = None,
    ) -> torch.Tensor:
        num_samples = num_samples or self.hparams.num_samples_to_generate_per_epoch
        diffusion_scale = diffusion_scale or self.hparams.diffusion_scale #should we use same diff scale?
        if num_negative_time_steps == -1:
            num_negative_time_steps = self.hparams.num_negative_time_steps

        if prior_samples is None:
            prior_samples = self.prior.sample(num_samples)

        samples, _ = self.integrate(
            reverse_sde=reverse_sde,
            samples=prior_samples.clone(),
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale, 
            resampling_interval=resampling_interval,
            num_langevin_steps=num_langevin_steps,
            batch_size=batch_size,
            num_negative_time_steps=num_negative_time_steps,
        )
        # TODO: When returning the weights for plotting, I am not doing additional langevin steps
        if return_logweights:
            # reintegrate without resampling to get logweights
            _, logweights = self.integrate(
                reverse_sde=reverse_sde,
                samples=prior_samples.clone(),
                reverse_time=True,
                return_full_trajectory=return_full_trajectory,
                diffusion_scale=diffusion_scale, 
                resampling_interval=self.num_integration_steps,
                num_langevin_steps=1,
                batch_size=batch_size,
                num_negative_time_steps=self.hparams.num_negative_time_steps, #TODO: Should we do any negative time here?
            )
            return samples, logweights
        return samples

    def eval_epoch_end(self, prefix: str):
        super().eval_epoch_end(prefix)

        if self.eval_count % 50 == 0:
            wandb_logger = get_wandb_logger(self.loggers)
            reverse_sde = VEReverseSDE(self.net, self.hparams.noise_schedule,
                                    inverse_temp=self.inverse_temp,
                                    scale_diffusion=self.hparams.scale_diffusion)

            prior_samples = self.annealed_prior.sample(self.hparams.num_eval_samples)

            self.last_samples_annealed, logweights = self.generate_samples(
                reverse_sde=reverse_sde,
                num_samples=self.hparams.num_eval_samples,
                return_logweights=True,
                resampling_interval=self.hparams.resampling_interval,
                diffusion_scale=self.hparams.diffusion_scale,  #TODO: what should the diffusion scale be?
                num_langevin_steps=self.hparams.num_langevin_steps,
                batch_size=self.hparams.num_samples_to_generate_per_epoch,
                prior_samples=prior_samples,
                num_negative_time_steps=self.hparams.num_negative_time_steps, #TODO: Should we do any negative time here?
            )
            self.last_energies_annealed = self.annealed_energy(self.last_samples_annealed)

            self.annealed_energy.log_on_epoch_end(
                self.last_samples_annealed,
                self.last_energies_annealed,
                wandb_logger,
                prefix="temp_annealed_samples",
            )

            self._log_energy_mean(self.last_energies_annealed, prefix="val/temp_annealed")

            if self.hparams.resampling_interval!=-1: #HERE
                self._log_logweights(logweights, prefix="val")
                self._plot_std_logweights(logweights, prefix="val")
            
        self.eval_count +=1


    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)
        super().on_test_epoch_end()

        # Compute metrics for the annealed energy
        batch_annealed_samples = sample_from_tensor(self.last_samples_annealed, self.hparams.eval_batch_size)
        batch_annealed_energies = self.annealed_energy(batch_annealed_samples)
        batch_test_samples = self.annealed_energy.sample_test_set(self.hparams.eval_batch_size)
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
        print("Energy W2", energy_w2)
        dist_w2 = pot.emd2_1d(
            self.annealed_energy.interatomic_dist(batch_annealed_samples).cpu().numpy().reshape(-1),
            self.annealed_energy.interatomic_dist(batch_test_samples).cpu().numpy().reshape(-1)
        )
        self.log(
            "test/temp_annealed/energy_w2",
            self.val_energy_w2(energy_w2),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        else:
            batch_size = self.num_samples_to_save
            n_batches = 1

        print(f"Generating {n_batches} batches of annealed samples of size {batch_size}.")

        reverse_sde = VEReverseSDE(self.net, self.hparams.noise_schedule,
                                   inverse_temp=self.inverse_temp,
                                   scale_diffusion=self.hparams.scale_diffusion)
        prior_samples = self.annealed_prior.sample(batch_size)

        for i in range(n_batches):
            start = time.time()
            samples = self.generate_samples(
                    reverse_sde=reverse_sde,
                    num_samples=batch_size,
                    return_logweights=False,
                    diffusion_scale=1.0,
                    resampling_interval=self.hparams.resampling_interval,
                    num_langevin_steps=self.hparams.num_langevin_steps,
                    batch_size=self.hparams.num_samples_to_generate_per_epoch,
                    prior_samples=prior_samples,
                    num_negative_time_steps=self.hparams.num_negative_time_steps

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
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
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

    def _plot_std_logweights(self, logweights, prefix="val"):
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

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.annealed_energy = self.hparams.annealed_energy(device=self.device)

        self.inverse_temp = self.energy_function.temperature / self.annealed_energy.temperature

        print("Inverse Temperature is", self.inverse_temp)
        
        self.annealed_prior = self.partial_prior(
            device=self.device, scale=(self.noise_schedule.h(1) / self.inverse_temp) ** 0.5
        )

        self.eval_count = 0


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