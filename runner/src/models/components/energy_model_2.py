from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction

class EnergyModel(nn.Module):
    def __init__(
        self, 
        score_net: nn.Module,
        target: BaseEnergyFunction,
        prior,
        noise_schedule,
    ):
        super(EnergyModel, self).__init__()
        self.score_net = score_net
        self.prior = prior
        self.target = target
        self.noise_schedule = noise_schedule

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # parametrize energy as <score_net(t, x), x>
        U_theta = 1/ (self.noise_schedule.h(t) ** 0.5) * torch.linalg.norm(self.score_net(t, x) - x, dim=-1)
        U_1 = self.prior.log_prob(x)
        # U_0 = self.target(x)
        return t * U_1 + (1 - t) * U_theta

    # def forward_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #     return self.score_net(t, x)