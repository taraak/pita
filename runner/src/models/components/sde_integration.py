import numpy as np
import torch
from src.models.components.sdes import VEReverseSDE
from src.models.components.utils import sample_cat_sys
from src.utils.data_utils import remove_mean
from src.energies.base_energy_function import BaseEnergyFunction
from contextlib import contextmanager


# def euler_maruyama_step(
#     sde: VEReverseSDE, t: torch.Tensor, x: torch.Tensor, dt: float, diffusion_scale=1.0
# ):
#     # Calculate drift and diffusion terms
#     drift = sde.f(t, x) * dt
#     diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x)

#     # Update the state
#     x_next = x + drift + diffusion
#     return x_next, drift

@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield

def grad_E(x, energy_function):
    with torch.enable_grad():
        x = x.requires_grad_()
        return torch.autograd.grad(torch.sum(energy_function(x)), x)[0].detach()

def negative_time_descent(x, energy_function, num_steps, dt=1e-4):
    samples = []
    for _ in range(num_steps):
        drift = grad_E(x, energy_function)
        x = x + drift * dt

        if energy_function.is_molecule:
            x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)

        samples.append(x)
    return torch.stack(samples)

def euler_maruyama_step(
        sde: VEReverseSDE,
        t: torch.Tensor,
        x: torch.Tensor,
        a: torch.tensor, 
        dt: float,
        step: int,
        batch_size: int,
        resampling_interval=-1,
        diffusion_scale=1.0,
        noise_correct=False
):
    # Calculate drift and diffusion terms for num_eval_batches
    EPS = 1e-6
    if noise_correct:
        sigma = np.sqrt(sde.g(0, x)**2 *  dt) - EPS
    else:
        sigma = 0.0

    drift_Xt = torch.zeros_like(x)
    drift_At = torch.zeros_like(a)

    x_noisy = x + sigma * torch.randn_like(x)

    for i in range(x.shape[0]//batch_size):
        drift_Xt_i, drift_At_i = sde.f(t, x_noisy[i*batch_size:(i+1)*batch_size], dt, 
                                       resampling_interval)
        drift_Xt[i*batch_size:(i+1)*batch_size] = drift_Xt_i
        drift_At[i*batch_size:(i+1)*batch_size] = drift_At_i


    diffusion = sde.diffusion(t, x, dt, diffusion_scale)

    # Update the state
    x_next = x + drift_Xt + diffusion
    a_next = a + drift_At

    if resampling_interval==-1 or step+1 % resampling_interval != 0:
        return x_next, a_next

    #resample based on the weights
    if a_next.isnan().any():
        print("NAN in the AIS weights")
    if torch.exp(-a_next).clamp(0, 10).isnan().any():
        print("NAN in the AIS weights exp")
    choice, _ = sample_cat_sys(x.shape[0], a_next)
    x_next = x_next[choice]
    a_next = torch.zeros_like(a_next)
    
    return x_next, a_next


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
    energy_function: BaseEnergyFunction,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    no_grad=True,
    time_range=1.0,
    resampling_interval=-1,
    num_negative_time_steps=100,
    num_langevin_steps=1,
    batch_size=None,
    noise_correct=False,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    if batch_size is None:
        batch_size = x0.shape[0]

    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x0.device
    )[:-1]
    x = x0

    x0.requires_grad = True
    samples = []
    logweights = []
    a = torch.zeros(x.shape[0]).to(x.device)

    with conditional_no_grad(no_grad):
        for step, t in enumerate(times):
            for _ in range(num_langevin_steps):
                x, a = euler_maruyama_step(sde, t, x, a, 
                                        time_range/num_integration_steps, step,
                                        resampling_interval=resampling_interval,
                                        diffusion_scale=diffusion_scale,
                                        batch_size=batch_size,
                                        noise_correct=noise_correct
                                        )
                if energy_function.is_molecule:
                    x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
                samples.append(x)
                logweights.append(a)

    samples = torch.stack(samples)
    logweights = torch.stack(logweights)

    if num_negative_time_steps > 0:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples, logweights
