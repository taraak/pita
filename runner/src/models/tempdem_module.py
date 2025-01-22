from typing import Any, Optional
import torch
import PIL
from .dem_module import *
from src.models.components.sdes import VEReverseSDE


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
        noise_correct=False,
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
            noise_correct=noise_correct
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
                num_negative_time_steps=0,
                noise_correct=noise_correct
            )
            return samples, logweights
        return samples

    def eval_epoch_end(self, prefix: str):
        super().eval_epoch_end(prefix)

        wandb_logger = get_wandb_logger(self.loggers)
        reverse_sde = VEReverseSDE(self.net, self.hparams.noise_schedule,
                                   inverse_temp=self.inverse_temp,
                                   scale_diffusion=self.hparams.scale_diffusion)

        prior_samples = self.annealed_prior.sample(self.hparams.num_eval_samples)

        self.last_samples, logweights = self.generate_samples(
            reverse_sde=reverse_sde,
            num_samples=self.hparams.num_eval_samples,
            return_logweights=True,
            resampling_interval=self.hparams.resampling_interval,
            diffusion_scale=self.hparams.diffusion_scale,  #TODO: what should the diffusion scale be?
            num_langevin_steps=self.hparams.num_langevin_steps,
            batch_size=self.hparams.num_samples_to_generate_per_epoch,
            prior_samples=prior_samples,
            num_negative_time_steps=0
        )
        self.last_energies = self.energy_function(self.last_samples)

        self.annealed_energy.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
            prefix="temp_annealed_samples",
        )

        if self.hparams.resampling_interval!=-1: #HERE
            self._log_logweights(logweights, prefix="val")
            self._plot_std_logweights(logweights, prefix="val")


    def on_test_epoch_end(self) -> None:
        wandb_logger = get_wandb_logger(self.loggers)

        self.eval_epoch_end("test")
        self._log_energy_w2(prefix="test")
        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="test")
            self._log_dist_total_var(prefix="test")
        
        batch_size = self.hparams.num_eval_samples
        final_samples = []
        n_batches = self.num_samples_to_save // batch_size
        print("Generating samples")

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
                )

        final_samples = torch.cat(final_samples, dim=0)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        path = f"{output_dir}/samples_{self.num_samples_to_save}.pt"
        torch.save(final_samples, path)
        print(f"Saving samples to {path}")
        import os
        os.makedirs(self.energy_function.name, exist_ok=True)
        path2 = f"{self.energy_function.name}/samples_{self.hparams.version}_{self.num_samples_to_save}.pt"
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
        # axs.set_ylim(-logweights.min()-1, logweights[0].mean() + logweights[0].std())
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