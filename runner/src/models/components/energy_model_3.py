from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction

class EnergyModel(nn.Module):
    def __init__(
        self, 
        score_net: nn.Module,
        target: BaseEnergyFunction,
        prior,
        score_clipper=None
    ):
        super(EnergyModel, self).__init__()
        self.target = target
        if self.target.is_molecule:
            self.score_net = score_net(energy_function=target, 
                                       add_virtual=False,
                                       energy=False)
        else:
            self.score_net = score_net()
        self.prior = prior
        self.score_clipper = score_clipper

    def forward_energy(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # parametrize energy as <score_net(t, x), x>
        U_theta = torch.sum(self.score_net(t, x) * x, dim=-1)
        U_1 = self.prior.log_prob(x)
        # U_0 = self.target(x)
        return t * U_1 + (1 - t) * U_theta

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        U = self.forward_energy(t, x)
        nabla_U = torch.autograd.grad(U.sum(), x, create_graph=True)[0]

        if self.score_clipper is not None:
            nabla_U = self.score_clipper.clip_scores(nabla_U)

        return nabla_U
