from torch import nn
import torch
from src.energies.base_energy_function import BaseEnergyFunction
from typing import Optional


class EnergyNet(nn.Module):
    def __init__(self, score_net: nn.Module):
        super(EnergyNet, self).__init__()
        self.score_net = score_net
        self.c = nn.Parameter(torch.tensor(0.0))

    def forward_energy(
        self,
        h_t: torch.Tensor,
        x: torch.Tensor,
        beta: torch.Tensor,
        pin: Optional[bool] = False,
        energy_function: Optional[BaseEnergyFunction] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta = beta * torch.ones(x.shape[0]).to(x.device)
        beta = beta.unsqueeze(1)

        c_s = 1 / (1 + h_t)  # 1 / (1 + sigma^2)
        c_in = 1 / (1 + h_t) ** 0.5  # 1 / sqrt(1 + sigma^2)
        c_out = h_t**0.5 * c_in  # sigma / sqrt(1 + sigma^2)
        c_noise = (1 / 8) * torch.log(h_t)  # 1/4 ln(sigma)

        def f_theta(t, xt, beta):
            h_theta = self.score_net(t, xt, beta)
            # h_theta = self.score_net(t, xt)
            return torch.sum(h_theta * xt, dim=1)

        U_theta = f_theta(c_noise, c_in[:, None] * x, beta)

        E_theta = (1 - c_s) / (2 * h_t) * torch.linalg.norm(x, dim=-1) ** 2 - c_out / (
            c_in * h_t
        ) * U_theta

        if pin:
            assert t is not None
            assert energy_function is not None
            U_0 = -energy_function(x)
            U_0 = torch.clamp(U_0, max=1e3, min=-1e3)
            return (1 - t) ** 3 * U_0 + (1 - (1 - t) ** 3) * E_theta
        return E_theta

    def forward(
        self, h_t: torch.Tensor, x: torch.Tensor, beta, pin=False, t=None
    ) -> torch.Tensor:
        U = self.forward_energy(h_t, x, beta, pin, t=t)
        nabla_U = torch.autograd.grad(U.sum(), x, create_graph=True)[0]
        return nabla_U