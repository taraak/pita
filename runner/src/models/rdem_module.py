from typing import Any, Optional
import torch
import PIL
from .components.energy_model_nem import EnergyModel
from .dem_module import *
from src.models.components.sdes import VEReverseSDE


class rDEMLitModule(DEMLitModule):
    def __init__(
        self,
        num_samples_to_generate_per_epoch_energy,
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
        do_langevin=False,
        resampling_interval=None,
        return_logweights=False,
    ) -> torch.Tensor:
        num_samples = num_samples or self.hparams.num_samples_to_generate_per_epoch_energy
        resampling_interval = resampling_interval or self.hparams.resampling_interval
        diffusion_scale = diffusion_scale or self.hparams.diffusion_scale #should we use same diff scale?

        prior_samples = self.prior.sample(num_samples)

        samples, _ = self.integrate(
            reverse_sde=reverse_sde,
            samples=prior_samples.clone(),
            reverse_time=True,
            return_full_trajectory=return_full_trajectory,
            diffusion_scale=diffusion_scale, 
            resampling_interval=resampling_interval
        )
        if return_logweights:
            # reintegrate without resampling to get logweights
            _, logweights = self.integrate(
                reverse_sde=reverse_sde,
                samples=prior_samples.clone(),
                reverse_time=True,
                return_full_trajectory=return_full_trajectory,
                diffusion_scale=diffusion_scale, 
                resampling_interval=self.num_integration_steps
            )
            return samples, logweights
        return samples

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
        with torch.enable_grad():
            samples.requires_grad_(True)
            predicted_score = self.net(times, samples)
            predicted_score_from_energy = self.energy_net(times, samples)

        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)
        loss_score_net = self.lambda_weighter(times) * error_norms

        error_norms_energy = (predicted_score_from_energy - predicted_score.detach()).pow(2).mean(-1)
        loss_energy_net = self.lambda_weighter(times) * error_norms_energy

        return loss_score_net + loss_energy_net

    def eval_epoch_end(self, prefix: str):
        super().eval_epoch_end(prefix)

        wandb_logger = get_wandb_logger(self.loggers)
        reverse_sde = VEReverseSDE(self.energy_net, self.noise_schedule, exact_hessian=self.hparams.exact_hessian)

        self.last_samples, logweights = self.generate_samples(
            reverse_sde=reverse_sde,
            return_logweights=True, #HERE
            resampling_interval=self.hparams.resampling_interval
        )
        self.last_energies = self.energy_function(self.last_samples)

        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
            prefix="energy_net",
        )


        if self.hparams.resampling_interval!=-1: #HERE
            self._log_logweights(logweights, prefix="val")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        #HERE changed to generate samples with og  net to see if that's the issue
        # reverse_sde = VEReverseSDE(self.energy_net, self.noise_schedule, exact_hessian=self.hparams.exact_hessian)
        # self.last_samples, logweights = self.generate_samples(
        #     reverse_sde=reverse_sde,
        #     return_logweights=True
        # )
        self.last_samples = self.generate_samples(resampling_interval=-1)
        
        self.last_energies = self.energy_function(self.last_samples)
        self.buffer.add(self.last_samples, self.last_energies)

        self._log_energy_w2(prefix="val")

        # if self.hparams.resampling_interval!=-1: #HERE
        #     self._log_logweights(logweights, prefix="val")

        if self.energy_function.is_molecule:
            self._log_dist_w2(prefix="val")
            self._log_dist_total_var(prefix="val")


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
            
    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.energy_net = EnergyModel(
            self.hparams.net,
            self.energy_function,
            self.prior,
            score_clipper=self.clipper_gen,
            pinned=self.hparams.pin_energy_net,
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