from typing import Optional

import torch
from src.energies.base_energy_function import BaseEnergyFunction
from torch import nn


class EnergyNet(nn.Module):
    def __init__(self, score_net: nn.Module, precondition_beta: Optional[bool] = False):
        super().__init__()
        self.net = score_net
        self.c = nn.Parameter(torch.tensor(0.0))
        self.precondition_beta = precondition_beta

    def forward_energy(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        beta: torch.Tensor,
        pin: Optional[bool] = False,
        energy_function: Optional[BaseEnergyFunction] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta = beta * torch.ones(x_t.shape[0]).to(x_t.device)

        c_s = 1 / (1 + h_t)  # 1 / (1 + sigma^2)
        c_in = 1 / (1 + h_t) ** 0.5  # 1 / sqrt(1 + sigma^2)
        c_out = h_t**0.5 * c_in  # sigma / sqrt(1 + sigma^2)
        c_noise = (1 / 8) * torch.log(h_t)  # 1/4 ln(sigma)

        def f_theta(t, x_t, beta):
            h_theta = self.net(t, x_t, beta)
            return torch.sum(h_theta * x_t, dim=1)

        U_theta = f_theta(c_noise, c_in[:, None] * x_t, beta)

        E_theta = (1 - c_s) / (2 * h_t) * torch.linalg.norm(x_t, dim=-1) ** 2 - c_out / (
            c_in * h_t
        ) * U_theta

        if self.precondition_beta:
            E_theta = E_theta * beta

        if pin:
            assert t is not None
            assert energy_function is not None
            U_0 = -energy_function(x_t)
            U_0 = torch.clamp(U_0, max=1e3, min=-1e3)
            return (1 - t) ** 3 * U_0 + (1 - (1 - t) ** 3) * E_theta
        return E_theta

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        beta: torch.Tensor,
        pin: Optional[bool] = False,
        t: Optional[torch.Tensor] = None,
        energy_function: Optional[BaseEnergyFunction] = None,
    ) -> torch.Tensor:
        U = self.forward_energy(h_t, x_t, beta, pin=pin, t=t, energy_function=energy_function)
        nabla_U = torch.autograd.grad(U.sum(), x_t, create_graph=True)[0]
        return nabla_U

    def denoiser(self, h_t: torch.Tensor, x_t: torch.Tensor, beta) -> torch.Tensor:
        nabla_U = self.forward(h_t, x_t, beta)
        return x_t - h_t[:, None] * nabla_U

    def denoiser_and_energy(self, h_t: torch.Tensor, x_t: torch.Tensor, beta) -> torch.Tensor:
        U = self.forward_energy(h_t, x_t, beta)
        nabla_U = torch.autograd.grad(U.sum(), x_t, create_graph=True)[0]
        return x_t - h_t[:, None] * nabla_U, U
