from contextlib import contextmanager

import numpy as np
import torch
from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.sdes import VEReverseSDE
from src.models.components.utils import sample_cat_sys
from src.utils.data_utils import remove_mean


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
            x = remove_mean(
                x, energy_function.n_particles, energy_function.n_spatial_dim
            )

        samples.append(x)
    return torch.stack(samples)


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
    x1: torch.Tensor,
    num_integration_steps: int,
    energy_function: BaseEnergyFunction,
    start_resampling_step: int,
    reverse_time: bool = True,
    diffusion_scale=1.0,
    time_range=1.0,
    resampling_interval=-1,
    inverse_temperature=1.0,
    annealing_factor=1.0,
    num_negative_time_steps=100,
    num_langevin_steps=1,
    batch_size=None,
    no_grad=True,
):
    start_time = time_range if reverse_time else 0.0
    end_time = time_range - start_time

    if batch_size is None:
        batch_size = x1.shape[0]

    times = torch.linspace(
        start_time, end_time, num_integration_steps + 1, device=x1.device
    )[:-1]
    x = x1

    x1.requires_grad = True
    samples = []
    logweights = []
    num_unique_idxs = []

    a = torch.zeros(x.shape[0], device=x.device)

    with conditional_no_grad(no_grad):
        for step, t in enumerate(times):
            for _ in range(num_langevin_steps):
                x, a, idxs = (
                    euler_maruyama_step(
                        sde,
                        t,
                        x,
                        a,
                        time_range / num_integration_steps,
                        step,
                        resampling_interval=resampling_interval,
                        diffusion_scale=diffusion_scale,
                        batch_size=batch_size,
                        start_resampling_step=start_resampling_step,
                        inverse_temperature=inverse_temperature,
                        annealing_factor=annealing_factor,
                        energy_function=energy_function,
                    )
                )
                if energy_function.is_molecule:
                    x = remove_mean(
                        x, energy_function.n_particles, energy_function.n_spatial_dim
                    )
                samples.append(x)
                logweights.append(a)
                num_unique_idxs.append(idxs)

    samples = torch.stack(samples)
    logweights = torch.stack(logweights)

    if num_negative_time_steps > 0:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x,
            energy_function,
            num_steps=num_negative_time_steps,
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples, logweights, num_unique_idxs


def euler_maruyama_step(
    sde: VEReverseSDE,
    t: torch.Tensor,
    x: torch.Tensor,
    a: torch.tensor,
    dt: float,
    step: int,
    batch_size: int,
    start_resampling_step: int,
    resampling_interval: int,
    inverse_temperature: float,
    annealing_factor: float,
    energy_function: BaseEnergyFunction,
    diffusion_scale: float,
):
    # Calculate drift and diffusion terms for num_eval_batches
    # drift_xt = torch.zeros_like(x)
    # drift_at = torch.zeros_like(a)
    # diffusion = torch.zeros_like(x)

    diffusion = []
    drift_at = []
    drift_xt = []


    # check time
    import time
    for i in range(x.shape[0] // batch_size):
        # t_start = time.time()
        drift_xt_i, drift_at_i = sde.f(
            t,
            x[i * batch_size : (i + 1) * batch_size],
            resampling_interval=resampling_interval,
            beta=inverse_temperature,
            gamma=annealing_factor,
            energy_function=energy_function,
        )
        # t_end = time.time()
        # print(f"Time taken for drift calculation: {t_end - t_start}")

        diffusion_i = sde.diffusion(
            t, x[i * batch_size : (i + 1) * batch_size], diffusion_scale
        )
        diffusion.append(diffusion_i)
        drift_xt.append(drift_xt_i)
        drift_at.append(drift_at_i)
        
    if x.shape[0] % batch_size != 0:
        i = x.shape[0] // batch_size
        drift_xt_i, drift_at_i = sde.f(
            t,
            x[i * batch_size :],
            resampling_interval=resampling_interval,
            beta=inverse_temperature,
            gamma=annealing_factor,
            energy_function=energy_function,
        )
        diffusion_i = sde.diffusion(
            t, x[i * batch_size :], diffusion_scale
        )
        diffusion.append(diffusion_i)
        drift_xt.append(drift_xt_i)
        drift_at.append(drift_at_i)

    diffusion = torch.cat(diffusion, dim=0)
    drift_xt = torch.cat(drift_xt, dim=0)
    drift_at = torch.cat(drift_at, dim=0)

    # update x, log weights, and log density
    dx = drift_xt * dt + diffusion * np.sqrt(dt)
    x_next = x + dx
    
    a_next = a + drift_at * dt

    # don't start accumulating weights until step start_resampling_step
    if step < start_resampling_step:
        a_next = torch.zeros_like(a_next)
        x_next = x  # samples are disytributed according to the prior, don't move

    if (
        resampling_interval == -1
        or (step + 1) % resampling_interval != 0
        or step < start_resampling_step
    ):
        return x_next, a_next, len(x_next)


    # resample based on the weights
    choice, _ = sample_cat_sys(x.shape[0], a_next)
    x_next = x_next[choice]
    a_next = torch.zeros_like(a_next)

    num_unique_idxs = len(np.unique(choice))

    return x_next, a_next, num_unique_idxs