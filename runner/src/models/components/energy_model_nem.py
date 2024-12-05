from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction

class EnergyModel(nn.Module):
    def __init__(
        self, 
        score_net: nn.Module,
        target: BaseEnergyFunction,
        prior,
        score_clipper=None,
        pinned=False
    ):
        super(EnergyModel, self).__init__()
        self.energy_function = target
        if self.energy_function.is_molecule:
            self.score_net = score_net(energy_function=target, 
                                       add_virtual=False,
                                       energy=True)
        else:
            self.score_net = score_net
            self.c = nn.Parameter(torch.tensor(0.0))
        self.prior = prior
        self.score_clipper = score_clipper
        self.pinned = pinned

    def forward_energy(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        score = self.score_net(t, x)
        U_1 = self.prior.log_prob(x)

        if not self.energy_function.is_molecule:
            U = - torch.linalg.vector_norm(score, dim=-1) + score.sum(-1) + self.c
            U_theta = U.unsqueeze(-1)
        
        else:
            score, potential = score
            score = score.view(score.shape[0], 
                            self.energy_function.n_particles,
                            self.energy_function.n_spatial_dim)
            potential = potential.view(score.shape[0], 
                            self.energy_function.n_particles, -1)
            U = - torch.linalg.vector_norm(score, dim=-1) + potential.sum(-1)
            U_theta = U.sum(-1)
        if self.pinned:
            return t * U_1 + (1 - t) * U_theta
        return U_theta


    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        U = self.forward_energy(t, x)
        nabla_U = torch.autograd.grad(U.sum(), x, create_graph=True)[0]

        if self.score_clipper is not None:
            nabla_U = self.score_clipper.clip_scores(nabla_U)

        return nabla_U