import numpy as np
import torch
from src.models.components.sdes import VEReverseSDE
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



def sample_cat_sys(bs, logits):
    u = torch.rand(size=(1,), dtype=torch.float64)
    u = (u + 1/bs*torch.arange(bs)) % 1.0
    clipped_weights = torch.clip(torch.softmax(logits, dim=-1), 1e-6, 1.0)
    bins = torch.cumsum(clipped_weights, dim=-1)
    ids = np.digitize(u, bins.cpu())
    return ids, None


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
):
    # Calculate drift and diffusion terms for num_eval_batches

    drift_Xt = torch.zeros_like(x)
    drift_At = torch.zeros_like(a)

    for i in range(x.shape[0]//batch_size):
        drift_Xt_i, drift_At_i = sde.f(t, x[i*batch_size:(i+1)*batch_size], resampling_interval)
        drift_Xt[i*batch_size:(i+1)*batch_size] = drift_Xt_i
        drift_At[i*batch_size:(i+1)*batch_size] = drift_At_i

    # drift_Xt, drift_At = sde.f(t, x, resampling_interval)
    drift_Xt = drift_Xt * dt
    drift_At = drift_At * dt

    if t.dim() == 0:
        # repeat the same time for all points if we have a scalar time
        t = t * torch.ones_like(x).to(x.device)
    diffusion = diffusion_scale * sde.g(t, x) * np.sqrt(dt) * torch.randn_like(x).to(x.device)

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
    # a_next = a_next.clamp(-10, 10)
    # choice = torch.multinomial(torch.softmax(-a_next, dim=-1), x.shape[0], replacement=True)
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
    negative_time=False,
    num_negative_time_steps=100,
    num_langevin_steps=1,
    batch_size=None,
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
                                        batch_size=batch_size)
                if energy_function.is_molecule:
                    x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
                samples.append(x)
                logweights.append(a)

    samples = torch.stack(samples)
    logweights = torch.stack(logweights)

    if negative_time:
        print("doing negative time descent...")
        samples_langevin = negative_time_descent(
            x, energy_function, num_steps=num_negative_time_steps
        )
        samples = torch.concatenate((samples, samples_langevin), axis=0)

    return samples, logweights



# def integrate_sde(
#     sde: VEReverseSDE,
#     x0: torch.Tensor,
#     num_integration_steps: int,
#     energy_function: BaseEnergyFunction,
#     reverse_time: bool = True,
#     diffusion_scale=1.0,
#     no_grad=True,
#     time_range=1.0,
# ):
#     start_time = time_range if reverse_time else 0.0
#     end_time = time_range - start_time

#     times = torch.linspace(
#         start_time, end_time, num_integration_steps + 1, device=x0.device
#     )[:-1]

#     x = x0
#     samples = []
#     if no_grad:
#         with torch.no_grad():
#             for t in times:
#                 x, f = euler_maruyama_step(
#                     sde, t, x, time_range / num_integration_steps, diffusion_scale
#                 )
#                 if energy_function.is_molecule:
#                     x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
#                 samples.append(x)
#     else:
#         for t in times:
#             x, f = euler_maruyama_step(
#                 sde, t, x, time_range / num_integration_steps, diffusion_scale
#             )
#             if energy_function.is_molecule:
#                 x = remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
#             samples.append(x)

#     return torch.stack(samples)