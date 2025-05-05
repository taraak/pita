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
        ht: torch.Tensor,
        xt: torch.Tensor,
        beta: torch.Tensor,
        pin: Optional[bool] = False,
        energy_function: Optional[BaseEnergyFunction] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta = beta * torch.ones(xt.shape[0]).to(xt.device)

        c_s = 1 / (1 + ht)  # 1 / (1 + sigma^2)
        c_in = 1 / (1 + ht) ** 0.5  # 1 / sqrt(1 + sigma^2)
        c_out = ht**0.5 * c_in  # sigma / sqrt(1 + sigma^2)
        c_noise = (1 / 8) * torch.log(ht)  # 1/4 ln(sigma)

        def f_theta(t, x_t, beta):
            h_theta = self.net(t, x_t, beta)
            return torch.sum(h_theta * x_t, dim=1)

        U_theta = f_theta(c_noise, c_in[:, None] * xt, beta)

        E_theta = (1 - c_s) / (2 * ht) * torch.linalg.norm(xt, dim=-1) ** 2 - c_out / (
            c_in * ht
        ) * U_theta

        if self.precondition_beta:
            E_theta = E_theta * beta

        if pin:
            assert t is not None
            assert energy_function is not None
            U_0 = -energy_function(xt)
            U_0 = torch.clamp(U_0, max=1e3, min=-1e3)
            return (1 - t) ** 3 * U_0 + (1 - (1 - t) ** 3) * E_theta
        return E_theta

    def forward(
        self,
        ht: torch.Tensor,
        xt: torch.Tensor,
        beta: torch.Tensor,
        pin: Optional[bool] = False,
        t: Optional[torch.Tensor] = None,
        energy_function: Optional[BaseEnergyFunction] = None,
    ) -> torch.Tensor:
        U = self.forward_energy(ht, xt, beta, pin=pin, t=t, energy_function=energy_function)       
        nabla_U = torch.autograd.grad(U.sum(), xt, create_graph=True)[0]
        return nabla_U

    def denoiser(self, h_t: torch.Tensor, x_t: torch.Tensor, beta) -> torch.Tensor:
        nabla_U = self.forward(h_t, x_t, beta)
        return x_t - h_t[:, None] * nabla_U

    def denoiser_and_energy(
            self,
            ht: torch.Tensor,
            xt: torch.Tensor,
            beta: torch.Tensor,
            # t: torch.Tensor,
    ) -> torch.Tensor:
        U = self.forward_energy(ht, xt, beta)
        U_grads = torch.autograd.grad(U.sum(), (xt, ht), create_graph=True)
        nabla_U = U_grads[0]
        dU_dt = U_grads[1]
        return xt - ht[:, None] * nabla_U, dU_dt, U
