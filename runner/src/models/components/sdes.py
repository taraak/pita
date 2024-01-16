import torch


class VEReverseSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule):
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        score = self.score(t, x)
        return self.g(t, x).pow(2) * score

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


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

class PIS_SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, tcond, noise_coeff, energy):
        super().__init__()
        self.score = score
        self.tcond = tcond
        self.noise_coeff = noise_coeff
        self.energy = energy

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        grad_log_energy = self.energy.score(x[..., :-1]).detach().to(x.device)
        dx = torch.clip(self.score(t, x[..., :-1]), -1e4, 1e4) * self.noise_coeff + self.tcond(t[0]) * torch.clip(grad_log_energy, -1e4, 1e4)
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x):
        return torch.cat(
            [torch.ones_like(x[..., :-1]) * self.noise_coeff, torch.zeros_like(x[..., -1:])], dim=-1
        )