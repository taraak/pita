import numpy as np
import torch
from src.models.components.temperature_schedules import ConstantInvTempSchedule
from src.models.components.utils import compute_laplacian, compute_divergence_forloop


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)



class VEReverseSDE(torch.nn.Module):
    def __init__(
        self, energy_net, noise_schedule, score_net=None, pin_energy=False
    ):
        super().__init__()
        self.energy_net = energy_net
        self.score_net = score_net
        self.noise_schedule = noise_schedule
        self.pin_energy = pin_energy

    def f(self, t, x, beta, gamma, resampling_interval=-1):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)
            h_t = self.noise_schedule.h(t).to(x.device)

            nabla_Ut = self.energy_net(h_t, x, beta)

            if self.score_net is not None:
                model_out = self.score_net(h_t, x, beta)
                bt = -model_out * self.g(t).pow(2).unsqueeze(-1) / 2
            else:
                bt = -nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2

            drift_X = gamma * (-nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2 + bt)

            drift_A = torch.zeros(x.shape[0]).to(x.device)

            if resampling_interval == -1 or t[0] > 0.9:
                return drift_X.detach(), drift_A.detach()

            Ut = self.energy_net.forward_energy(
                h_t, x, beta, pin=self.pin_energy, t=t
            )

            if self.score_net is not None:
                div_bt = compute_divergence_forloop(
                    bt,
                    x,
                )
            else:
                laplacian_Ut = compute_divergence_forloop(
                    nabla_Ut,
                    x,
                )
                div_bt = -laplacian_Ut * (self.g(t).pow(2) / 2)

            dUt_dt = torch.autograd.grad(Ut.sum(), t, create_graph=True)[0]

            drift_A = (
                gamma**2 * (-nabla_Ut * bt).sum(-1)
                + gamma * div_bt
                + gamma * dUt_dt
            )

        return drift_X.detach(), drift_A.detach()

    def g(self, t):
        g = self.noise_schedule.g(t)
        return g
    

class RegVEReverseSDE(VEReverseSDE):
    def f(self, t, x):
        dx = super().f(t, x[..., :-1])
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        if g.ndim > 0:
            return g.unsqueeze(1)
        return torch.cat([torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1)
