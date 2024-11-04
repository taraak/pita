from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction

class EnergyModel(nn.Module):
    def __init__(
        self, 
        score_net: nn.Module,
        target: BaseEnergyFunction,
        prior
    ):
        super(EnergyModel, self).__init__()
        self.score_net = score_net
        self.prior = prior
        self.target = target

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # parametrize energy as <score_net(t, x), x>
        U_theta = torch.sum(self.score_net(t, x) * x, dim=1)
        U_1 = self.prior.log_prob(x)
        U_0 = self.target(x)

        return torch.sin(t) * U_theta + t * U_1 + (1-t) * U_0

    # def forward_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #     return self.score_net(t, x)