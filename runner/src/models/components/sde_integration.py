import numpy as np
import torch

from src.models.components.sdes import VEReverseSDE


def euler_maruyama_step(
    sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0
):
    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

    # Update the state
    x_next = x + drift + diffusion
    return x_next, drift


def integrate_pfode(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start_time

    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x0.device
    )[:-1]

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, t, x, 1 / num_integration_steps)
            samples.append(x)

    return torch.stack(samples)


def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x0.device
    )[:-1]

    x = x0
    samples = []
    if no_grad:
        with torch.no_grad():
            for t in times:
                x, f = euler_maruyama_step(
                    sde, t, x, time_range / num_integration_steps, diffusion_scale
                )
                samples.append(x)
    else:
        for t in times:
            x, f = euler_maruyama_step(
                sde, t, x, time_range / num_integration_steps, diffusion_scale
            )
            samples.append(x)

    return torch.stack(samples)
