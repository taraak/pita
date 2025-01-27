import torch
import numpy as np
from src.models.components.utils import compute_laplacian
from src.models.components.temperature_schedules import ConstantInvTempSchedule

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
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model, noise_schedule, temperature_schedule=None,
                 scale_diffusion=False, clipper=None):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.scale_diffusion = scale_diffusion
        self.clipper = clipper
        if temperature_schedule is None:
            self.temperature_schedule = ConstantInvTempSchedule(1.0)
        else:
            self.temperature_schedule = temperature_schedule
 
    def f(self, t, x, resampling_interval=None):
        beta_t = self.temperature_schedule.beta(t)
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)

            nabla_qt = self.model(t, x)
            self.nabla_logqt = nabla_qt

            if self.scale_diffusion:
                annealed_nabla_qt = 0.5 * (beta_t+1) * nabla_qt
            else:
                annealed_nabla_qt = beta_t * nabla_qt

            if self.clipper is not None and beta_t > 1:
                annealed_nabla_qt = self.clipper.clip_scores(annealed_nabla_qt)

            drift_X = annealed_nabla_qt * self.g(t, x).pow(2).unsqueeze(-1)
            drift_A = torch.zeros(x.shape[0]).to(x.device)
            
            if resampling_interval==-1:
                return  drift_X.detach(), drift_A.detach()
            
            drift_A = 0.5 * (beta_t - 1) * beta_t * (self.g(t, x)[:, None] * nabla_qt).pow(2).sum(-1)
            
        return  drift_X.detach(), drift_A.detach()

    
    def diffusion(self, t, x, diffusion_scale):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones_like(x).to(x.device)
        beta_t = self.temperature_schedule.beta(t)

        self.noise = torch.randn_like(x).to(x.device)

        diffusion = diffusion_scale * self.g(t, x) * self.noise
        if self.scale_diffusion:
            return  diffusion/ torch.sqrt(beta_t)
        return diffusion

    def g(self, t, x):
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
        return torch.cat(
            [torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1
        )
    