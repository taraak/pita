import torch
from torch.func import jacrev
from torchdiffeq import odeint


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)

    def div(x):
        return torch.trace(J(x.unsqueeze(0)).squeeze())

    return div


class CNF(torch.nn.Module):
    def __init__(self, vf, is_diffusion):
        super().__init__()

        self.vf = vf
        self.is_diffusion = is_diffusion
        self.nfe = 0.0

    def forward(self, t, x):
        x = x[..., :-1].clone().detach().requires_grad_(True)

        def vecfield(x):
            # PF ODE requires dividing the reverse (VE) drift by 2.
            # If we use VP (or a different noising SDE) we need the
            # forward drift too.
            shaped_t = torch.ones(x.shape[0], device=x.device) * t
            return self.vf(shaped_t, x) / (2.0 if self.is_diffusion else 1.0)

        dx = vecfield(x)
        div = torch.vmap(div_fn(vecfield), randomness="different")(x)
        self.nfe += 1
        return torch.cat([dx, div[:, None]], dim=-1)

    def integrate(self, x, num_integration_steps: int, method="euler"):
        end_time = int(self.is_diffusion)
        start_time = 1.0 - end_time

        time = torch.linspace(
            start_time, end_time, num_integration_steps + 1, device=x.device
        )

        return odeint(self, x, t=time, method=method)
