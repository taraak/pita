from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction
from typing import Optional


class ScoreNet(nn.Module):
    def __init__(self, model: nn.Module, x0: torch.Tensor):
        super(ScoreNet, self).__init__()

        self.model = model
        self.x0 = x0

    def forward(
        self, h_t: torch.Tensor, x_t: torch.Tensor, beta
    ) -> torch.Tensor:
        return (self.x0 - x_t) / h_t[:, None]
        # return (self.denoiser(h_t, x_t, beta) - x_t) / h_t[:, None]

    def denoiser(self, h_t: torch.Tensor, x_t: torch.Tensor, beta, return_score=False) -> torch.Tensor:
        beta = beta * torch.ones(x_t.shape[0]).to(x_t.device)

        c_s = 1 / (1 + h_t)  # 1 / (1 + sigma^2)
        c_in = 1 / (1 + h_t) ** 0.5  # 1 / sqrt(1 + sigma^2)
        c_out = h_t**0.5 * c_in  # sigma / sqrt(1 + sigma^2)
        c_noise = (1 / 8) * torch.log(h_t)  # 1/4 ln(sigma)


        D_theta =  x_t + self.model.forward(c_noise, x_t, beta)
        
        if return_score:
            score = (D_theta - x_t) / h_t[:, None]
            return D_theta, score
        
        return D_theta

