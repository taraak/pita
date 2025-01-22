import torch
import numpy as np
from src.models.components.utils import compute_laplacian

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

    def __init__(self, model, noise_schedule, scale_diffusion=False, inverse_temp=1.0):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.scale_diffusion = scale_diffusion
        self.inverse_temp = inverse_temp

    def f(self, t, x, dt, resampling_interval=None):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)

            nabla_Ut = self.model(t, x)

            if self.scale_diffusion:
                drift_X = 0.5 * (self.inverse_temp+1) * nabla_Ut * self.g(t, x).pow(2).unsqueeze(-1) * dt
            else:
                drift_X = self.inverse_temp * nabla_Ut * self.g(t, x).pow(2).unsqueeze(-1) * dt

            drift_A = torch.zeros(x.shape[0]).to(x.device)
            
            if resampling_interval is None:
                return  drift_X.detach(), drift_A.detach()
            
            drift_A = (0.5 * (self.inverse_temp-1) * self.inverse_temp 
                       * (self.g(t, x)[:, None] * nabla_Ut).pow(2).sum(-1) * dt)
                
        return  drift_X.detach(), drift_A.detach()
    

    def diffusion(self, t, x, dt, diffusion_scale, sigma=0.0):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones_like(x).to(x.device)

        if self.scale_diffusion:
            return (diffusion_scale * torch.sqrt(self.g(t, x) ** 2 * dt - sigma) * torch.randn_like(x).to(x.device) 
                    / np.sqrt(self.inverse_temp))
            # return diffusion_scale * self.g(t, x) * torch.randn_like(x).to(x.device) / np.sqrt(inverse_temp)
        
        # return diffusion_scale * self.g(t, x) * torch.randn_like(x).to(x.device)
        return diffusion_scale * torch.sqrt(self.g(t, x) ** 2 * dt - sigma) * torch.randn_like(x).to(x.device)

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
    