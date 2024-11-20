import torch
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

    def __init__(self, model, noise_schedule, exact_hessian=False):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.exact_hessian = exact_hessian

    def f(self, t, x,  resampling_interval=None):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)
            epsilon_t = self.g(t, x).pow(2) / 2 
            
            Ut = self.model.forward_energy(t, x)
            # nabla_Ut = torch.autograd.grad(Ut.sum(), x, create_graph=True)[0]
            nabla_Ut = self.model(t, x)

            drift_X = nabla_Ut * self.g(t, x).pow(2).unsqueeze(-1)

            drift_A = torch.zeros(x.shape[0]).to(x.device)

            if resampling_interval is None:
                return  drift_X, drift_A
            
            dUt_dt = torch.autograd.grad(Ut.sum(), t, create_graph=True)[0]

            laplacian_b = epsilon_t * compute_laplacian(self.model.forward_energy, nabla_Ut, t, x, 1, exact=self.exact_hessian)
            drift_A = -laplacian_b - epsilon_t * nabla_Ut.pow(2).sum(-1) + dUt_dt

        return  drift_X.detach(), drift_A.detach()

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g

# class VEReverseSDE(torch.nn.Module):
#     noise_type = "diagonal"
#     sde_type = "ito"

#     def __init__(self, score, noise_schedule):
#         super().__init__()
#         self.score = score
#         self.noise_schedule = noise_schedule

#     def f(self, t, x):
#         if t.dim() == 0:
#             # repeat the same time for all points if we have a scalar time
#             t = t * torch.ones(x.shape[0]).to(x.device)

#         score = self.score(t, x)
#         return self.g(t, x).pow(2) * score

#     def g(self, t, x):
#         g = self.noise_schedule.g(t)
#         return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


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