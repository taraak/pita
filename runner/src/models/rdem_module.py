from typing import Any, Dict, Optional
import time
import PIL
from hydra.utils import get_original_cwd
import hydra
import matplotlib.pyplot as plt
import ot as pot
import numpy as np
import torch

from .dem_module import *

class rDEMLitModule(DEMLitModule):

    def forward_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.score_net(t, x)


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
            predicted_score = self.forward_score(times, samples)
            predicted_score_from_energy = self.net(times, samples)
        
        error_norms = (predicted_score - estimated_score).pow(2).mean(-1)
        loss_score_net = self.lambda_weighter(times) * error_norms

        error_norms_energy = (predicted_score_from_energy - predicted_score.detach()).pow(2).mean(-1)
        loss_energy_net = self.lambda_weighter(times) * error_norms_energy

        return loss_score_net + loss_energy_net


    def eval_epoch_end(self, prefix: str):
        super().eval_epoch_end(prefix)

        wandb_logger = get_wandb_logger(self.loggers)
        
        reverse_sde = VEReverseSDE(self.score_net, self.noise_schedule, exact_hessian=self.hparams.exact_hessian)

        self.last_samples, _ = self.generate_samples(
            reverse_sde = reverse_sde,
            return_logweights=True
        )
        
        self.last_energies = self.energy_function(self.last_samples)

        self.energy_function.log_on_epoch_end(
            self.last_samples,
            self.last_energies,
            wandb_logger,
            prefix = "score_net",
            )


    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.score_net = self.hparams.net()



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
