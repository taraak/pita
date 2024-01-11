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
    def __init__(self, vf):
        super().__init__()
        self.vf = vf

    def forward(self, t, x):
        x = x[..., :-1].clone().detach().requires_grad_(True)

        def vecfield(x):
            return self.vf(torch.ones(x.shape[0], device=x.device) * t, x)

        dx = vecfield(x)
        div = torch.vmap(div_fn(vecfield), randomness="different")(x)
        return torch.cat([dx, div[:, None]], dim=-1)

    def integrate(self, x, num_integration_steps: int, method="euler"):
        time = torch.linspace(0, 1, num_integration_steps, device=x.device)
        return odeint(self, x, t=time, method=method)
