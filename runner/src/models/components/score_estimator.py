import torch
from src.data import BaseEnergyFunction
from src.models.components import BaseNoiseSchedule

def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int
):
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())
    log_rewards = energy_function(samples)

    return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)

def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int
):
    grad_fxn = torch.func.grad(log_expectation_reward)
    vmapped_fxn = torch.vmap(
        grad_fxn,
        in_dims=(0, 0, None, None, None),
        randomness='different'
    )

    return vmapped_fxn(
        t,
        x,
        energy_function,
        noise_schedule,
        num_mc_samples
    )
