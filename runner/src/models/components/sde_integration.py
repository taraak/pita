import torch
from src.models.components.sdes import VEReverseSDE

def euler_maruyama_step(
	sde: VEReverseSDE,
    t: torch.Tensor,
	x: torch.Tensor,
    dt: float
):
    # Calculate drift and diffusion terms
    drift = sde.f(t, x) * dt
    diffusion = sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x).to(device)

    # Update the state
    x_next = x + drift + diffusion
    return x_next, drift

def integrate_sde(
    sde: VEReverseSDE,
    x0: torch.Tensor,
    num_integration_steps: int,
    reverse_time: bool
):
    start_time = 1.0 if reverse_time else 0.0
    end_time = 1.0 - start

    times = torch.linspace(
        start_time,
        end_time,
        num_integration_steps,
        device=x0.device
    )

    x = x0
    samples = []
    with torch.no_grad():
        for t in times:
            x, f = euler_maruyama_step(sde, x, t, dt)
            samples.append(x)

    return torch.stack(samples)
