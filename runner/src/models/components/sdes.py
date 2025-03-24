import numpy as np
import torch
from src.models.components.temperature_schedules import ConstantInvTempSchedule
from src.models.components.utils import compute_laplacian_exact, compute_divergence_forloop, compute_divergence_exact


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
        self, energy_net, noise_schedule, score_net=None, pin_energy=False,
        debias_inference=True
    ):
        super().__init__()
        self.energy_net = energy_net
        self.score_net = score_net
        self.noise_schedule = noise_schedule
        self.pin_energy = pin_energy
        self.debias_inference = debias_inference


    def f_not_debiased(self, t, x, beta):
        assert self.score_net is not None
        h_t = self.noise_schedule.h(t)
        drift_X = self.score_net(h_t, x, beta) * self.g(t).pow(2).unsqueeze(-1) 
        drift_A = torch.zeros(x.shape[0]).to(x.device)
        return drift_X, drift_A
        

    def f(self, t, x, beta, gamma, energy_function, resampling_interval=-1):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)
        
                    
        if not self.debias_inference:
            return self.f_not_debiased(t, x, beta)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)
        
            h_t = self.noise_schedule.h(t)
            nabla_Ut = self.energy_net(h_t, x, beta)

            if self.score_net is not None:
                s_t = self.score_net(h_t, x, beta)
                bt = s_t * self.g(t).pow(2).unsqueeze(-1) / 2
            else:
                bt = -nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2

            drift_X = (gamma * (-nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2 + bt)).detach()

            drift_A = torch.zeros(x.shape[0]).to(x.device)

            if resampling_interval == -1:
                return drift_X.detach(), drift_A.detach()

            Ut = self.energy_net.forward_energy(
                h_t = h_t,
                x_t = x,
                beta = beta,
                pin = self.pin_energy,
                energy_function = energy_function,
                t = t,
            )

            if self.score_net is not None:
                div_st = compute_divergence_exact(
                    self.score_net,
                    t,
                    x,
                    beta,
                ).detach() 
                div_bt = div_st * self.g(t).pow(2) / 2
            else:
                # laplacian_Ut = compute_divergence_forloop(
                #     nabla_Ut,
                #     x,
                # ).detach()
                
                laplacian_Ut = compute_laplacian_exact(
                    self.energy_net,
                    t,
                    x,
                    beta,
                ).detach()
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


    def diffusion(self, t, x, diffusion_scale):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        diffusion = ( 
            diffusion_scale
            * self.g(t)[:, None]
            * torch.randn_like(x).to(x.device)
            )
        return diffusion

