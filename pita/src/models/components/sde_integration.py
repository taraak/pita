from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich.progress import Progress
from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.annealing_factor_schedules import BaseAnnealingFactorSchedule
from src.models.components.sdes import SDETerms, VEReverseSDE
from src.models.components.utils import sample_cat_sys
from src.utils.data_utils import remove_mean
from tqdm.auto import tqdm

disable = rank_zero_only.rank != 0


@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


def mala_proposal(x, energy_function, dt):
    grad = []
    _, grad = energy_function(x, return_force=True)

    noise = torch.randn_like(x)
    x_prop = x + 0.5 * dt * grad + torch.sqrt(torch.tensor(dt)) * noise

    # Forward proposal: q(x' | x)
    forward_mean = x + 0.5 * dt * grad
    log_q_forward = -((x_prop - forward_mean) ** 2).sum(dim=1) / (2 * dt)

    # Backward proposal: q(x | x')
    _, grad_prop = energy_function(x_prop, return_force=True)

    backward_mean = x_prop + 0.5 * dt * grad_prop
    log_q_backward = -((x.detach() - backward_mean) ** 2).sum(dim=1) / (2 * dt)

    return x_prop.detach(), log_q_forward.detach(), log_q_backward.detach()


class WeightedSDEIntegrator:
    def __init__(
        self,
        sde: VEReverseSDE,
        num_integration_steps: int,
        start_resampling_step: int,
        end_resampling_step: int,
        lightning_module: LightningModule,
        partial_annealing_factor_schedule: BaseAnnealingFactorSchedule,
        reverse_time: bool = True,
        diffusion_scale=1.0,
        time_range=1.0,
        resampling_interval=-1,
        num_negative_time_steps=100,
        post_mcmc_steps=100,
        adaptive_mcmc=False,
        batch_size=None,
        no_grad=True,
        resample_at_end=False,
        dt_negative_time=1e-4,
        do_langevin=False,
        should_mean_free=True,
    ) -> None:
        self.sde = sde
        self.num_integration_steps = num_integration_steps
        self.start_resampling_step = start_resampling_step
        self.end_resampling_step = end_resampling_step
        self.reverse_time = reverse_time
        self.diffusion_scale = diffusion_scale
        self.resampling_interval = resampling_interval
        self.time_range = time_range
        self.num_negative_time_steps = num_negative_time_steps
        self.post_mcmc_steps = post_mcmc_steps
        self.adaptive_mcmc = adaptive_mcmc
        self.dt_negative_time = dt_negative_time
        self.batch_size = batch_size
        self.no_grad = no_grad
        self.resample_at_end = resample_at_end
        self.do_langevin = do_langevin
        self.lightning_module = lightning_module
        self.should_mean_free = should_mean_free

        self.start_time = time_range if reverse_time else 0.0
        self.end_time = time_range - self.start_time

    def maybe_remove_mean(self, x, energy_function):
        if self.should_mean_free:
            return remove_mean(x, energy_function.n_particles, energy_function.n_spatial_dim)
        return x

    def integrate_sde(
        self,
        x1: torch.Tensor,
        energy_function: BaseEnergyFunction,
        annealing_factor_schedule: BaseAnnealingFactorSchedule,
        inverse_temperature=1.0,
        annealing_factor_score=1.0,
        resampling_interval=None,
    ):
        if self.batch_size is None:
            self.batch_size = x1.shape[0]

        if resampling_interval is None:
            resampling_interval = self.resampling_interval

        print("The resampling interval is: ", resampling_interval)

        times = torch.linspace(
            self.start_time,
            self.end_time,
            self.num_integration_steps + 1,
            device=x1.device,
        )[:-1]

        dt = self.time_range / self.num_integration_steps

        x = x1
        x1.requires_grad = True
        logweights = []
        num_unique_idxs = []
        sde_terms_all = []
        a = torch.zeros(x.shape[0], device=x.device)

        with conditional_no_grad(self.no_grad):
            for step, t in enumerate(
                tqdm(times, position=0, leave=True, dynamic_ncols=True, disable=disable)
            ):
                # for step, t in enumerate(times):
                x, a, idxs, sde_terms = self.ddp_batched_euler_maruyama_step(
                    t,
                    x,
                    a,
                    dt,
                    step,
                    inverse_temperature=inverse_temperature,
                    annealing_factor=annealing_factor_schedule,
                    annealing_factor_score=annealing_factor_score,
                    energy_function=energy_function,
                    resampling_interval=resampling_interval,
                )
                x = self.maybe_remove_mean(x, energy_function)

                logweights.append(a)
                num_unique_idxs.append(idxs)
                sde_terms_all.append(sde_terms)

                # # TODO: added this here
                # if step > self.end_resampling_step:
                #     break

            did_resampling = resampling_interval != -1 and resampling_interval < len(times)
            if self.resample_at_end and did_resampling:
                # t = torch.tensor(self.end_time).to(x.device)
                # TODO: changed this time
                t = times[self.end_resampling_step]
                print(f"doing end resampling at {t}")
                target_logprob = energy_function(x)
                if t.dim() == 0:
                    t = t * (torch.ones(x.shape[0])).to(x.device)
                h_t = self.sde.noise_schedule.h(t)
                model_energy = self.sde.energy_net.forward_energy(
                    h_t,
                    x,
                    inverse_temperature,
                    pin=self.sde.pin_energy,
                    energy_function=energy_function,
                    t=t,
                )
                logq_0 = -model_energy * annealing_factor_schedule.gamma(t)
                a_next = target_logprob - logq_0 + a
                # a_next = torch.clamp(a_next - a_next.mean(), max=10)
                a_next = torch.clamp(a_next, max=torch.quantile(a_next, 0.9))
                choice, _ = sample_cat_sys(x.shape[0], a_next)
                x = x[choice]
                logweights.append(a_next.detach())
                num_unique_idxs.append(len(np.unique(choice)))

        logweights = torch.stack(logweights)

        if self.num_negative_time_steps > 0:
            print(f"doing negative time descent for {self.num_negative_time_steps} steps")
            x = self.negative_time_descent(
                x,
                energy_function,
            )

        acceptance_rate_list = []
        if self.post_mcmc_steps > 0:
            if self.adaptive_mcmc:
                print(f"adaptive mcmc for {self.post_mcmc_steps} steps")
                x, acceptance_rate_list = self.metropolis_hastings_mala_adaptive(
                    x,
                    energy_function,
                    dt_init=self.dt_negative_time,
                    return_acceptance_rate=True,
                )
            else:
                print(f"mcmc for {self.post_mcmc_steps} steps")
                x, acceptance_rate_list = self.metropolis_hastings_mala(
                    x,
                    energy_function,
                    return_acceptance_rate=True,
                )

        return x, logweights, num_unique_idxs, sde_terms_all, acceptance_rate_list

    def ddp_batched_euler_maruyama_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        a: torch.tensor,
        dt: float,
        step: int,
        inverse_temperature: float,
        annealing_factor: BaseAnnealingFactorSchedule,
        annealing_factor_score: float,
        energy_function: BaseEnergyFunction,
        resampling_interval: int,
    ):
        local_batch_size = x.shape[0] // self.lightning_module.trainer.world_size

        rank = self.lightning_module.trainer.global_rank

        # split x and a into batches
        x = x[rank * local_batch_size : (rank + 1) * local_batch_size]
        a = a[rank * local_batch_size : (rank + 1) * local_batch_size]

        x_next, a_next, sde_terms = self.euler_maruyama_step(
            t,
            x,
            a,
            dt,
            inverse_temperature,
            annealing_factor,
            annealing_factor_score,
            energy_function,
            resampling_interval,
        )

        # gather x and a
        x_next = self.lightning_module.all_gather(x_next).reshape(-1, *x_next.shape[1:])
        a_next = self.lightning_module.all_gather(a_next).reshape(-1, *a_next.shape[1:])

        sde_terms = self.lightning_module.all_gather(asdict(sde_terms))
        for term in fields(SDETerms):
            if sde_terms[term.name] is None:
                continue
            sde_terms[term.name] = sde_terms[term.name].reshape(
                -1, *sde_terms[term.name].shape[2:]
            )
        sde_terms = SDETerms(**sde_terms)

        if False:  # resampling_interval == 1:
            import os

            path = "/scratch/t/taraak/energy_temp/"
            os.makedirs(path + "debug", exist_ok=True)
            torch.save(
                {
                    "x_next": x_next,
                    "a_next": a_next,
                    "sde_terms": SDETerms.cpu(sde_terms),
                    "step": step,
                    "t": t,
                },
                f"{path}/debug/step_{step}.pt",
            )
            print(f"saved step {step} to {path}/debug/step_{step}.pt")

        # don't start accumulating weights until step start_resamplings_step
        if step < self.start_resampling_step:
            a_next = torch.zeros_like(a_next)
            x_next = x  # samples are distributed according to the prior, don't move
        if step >= self.end_resampling_step:
            a_next = torch.zeros_like(a_next)
        if (
            resampling_interval == -1
            or (step + 1) % resampling_interval != 0
            or step < self.start_resampling_step
            or step >= self.end_resampling_step
        ):
            return x_next, a_next, len(x_next), SDETerms.cpu(sde_terms)

        # resample based on the weights
        choice, _ = sample_cat_sys(x_next.shape[0], a_next)
        x_next = x_next[choice]
        a_next = torch.zeros_like(a_next)
        num_unique_idxs = len(np.unique(choice))

        return x_next, a_next, num_unique_idxs, SDETerms.cpu(sde_terms)

    def euler_maruyama_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        a: torch.tensor,
        dt: float,
        inverse_temperature: float,
        annealing_factor: BaseAnnealingFactorSchedule,
        annealing_factor_score: float,
        energy_function: BaseEnergyFunction,
        resampling_interval: int,
    ):
        sde_terms = []
        for i in range(x.shape[0] // self.batch_size):
            sde_term = self.sde.f(
                t,
                x[i * self.batch_size : (i + 1) * self.batch_size],
                resampling_interval=resampling_interval,
                beta=inverse_temperature,
                gamma_energy_schedule=annealing_factor,
                gamma_score=annealing_factor_score,
                energy_function=energy_function,
            )
            sde_term.diffusion = self.sde.diffusion(
                t,
                x[i * self.batch_size : (i + 1) * self.batch_size],
                self.diffusion_scale,
            )
            sde_terms.append(sde_term)

        if x.shape[0] % self.batch_size != 0:
            i = x.shape[0] // self.batch_size
            sde_term = self.sde.f(
                t,
                x[i * self.batch_size :],
                resampling_interval=resampling_interval,
                beta=inverse_temperature,
                gamma_energy_schedule=annealing_factor,
                gamma_score=annealing_factor_score,
                energy_function=energy_function,
            )
            sde_term.diffusion = self.sde.diffusion(
                t, x[i * self.batch_size :], self.diffusion_scale
            )
            sde_terms.append(sde_term)
        sde_terms = SDETerms.concatenate(sde_terms)

        # update x, log weights, and log density
        dx = sde_terms.drift_X * dt + sde_terms.diffusion * np.sqrt(dt)
        x_next = x + dx
        a_next = a + sde_terms.drift_A * dt

        return x_next, a_next, sde_terms

    def negative_time_descent(self, x, energy_function):
        for _ in range(self.num_negative_time_steps):
            _, drift = energy_function(x, return_force=True)
            x = x + drift * self.dt_negative_time
            if self.do_langevin:
                x = x + torch.randn_like(x) * np.sqrt(2 * self.dt_negative_time)
            x = self.maybe_remove_mean(x, energy_function)
        return x

    def metropolis_hastings_mala(self, x, energy_function, return_acceptance_rate=False):
        """updated MALA with filtering"""
        x_curr = x.clone()
        logp_curr = energy_function(x_curr)
        valid_mask = torch.isfinite(logp_curr).squeeze()
        # print(f"number of valid samples:{valid_mask.sum().item()} out of {x_curr.size(0)} samples")
        x_valid = x_curr[valid_mask]
        x_invalid = x_curr[~valid_mask]
        acceptance_rate_list = []
        for i in range(self.post_mcmc_steps):
            try:
                # Propose updates for valid rows
                if x_valid.size(0) > 0:
                    x_valid_prop, log_q_forward, log_q_backward = mala_proposal(
                        x_valid, energy_function, self.dt_negative_time
                    )
                    logp_valid_prop = energy_function(x_valid_prop)
                    log_accept_ratio = (logp_valid_prop - logp_curr[valid_mask]) + (
                        log_q_backward - log_q_forward
                    )
                    # Accept or reject the proposal for valid rows
                    accept_mask = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
                    accept_mask_float = accept_mask.float()
                    accept_valid = accept_mask_float.unsqueeze(-1)
                    if return_acceptance_rate:
                        acceptance_rate = accept_mask.float().mean().item()
                        acceptance_rate_list.append(acceptance_rate)
                    print(f"{i}th MCMC acceptance rate: {acceptance_rate}")

                    # Update valid rows based on acceptance
                    x_valid = accept_valid * x_valid_prop + (1 - accept_valid) * x_valid
                    logp_curr[valid_mask] = (
                        accept_mask_float * logp_valid_prop
                        + (1 - accept_mask_float) * logp_curr[valid_mask]
                    )
                    if energy_function.is_molecule:
                        x_valid = self.maybe_remove_mean(x_valid, energy_function)

                    x_curr = torch.cat([x_valid, x_invalid], dim=0)
            except Exception as e:
                print(f"Error during MALA step: {e}")

        if return_acceptance_rate:
            return x_curr, acceptance_rate_list
        else:
            return x_curr, None

    """ add adaptive steps """

    def metropolis_hastings_mala_adaptive(
        self, x, energy_function, dt_init, return_acceptance_rate=False
    ):
        x_curr = x.clone()
        logp_curr = energy_function(x_curr)
        valid_mask = torch.isfinite(logp_curr).squeeze()

        x_valid = x_curr[valid_mask]
        x_invalid = x_curr[~valid_mask]
        dt = dt_init
        acceptance_rate_list = []
        for i in range(self.post_mcmc_steps):
            try:
                if x_valid.size(0) > 0:
                    x_valid_prop, log_q_forward, log_q_backward = mala_proposal(
                        x_valid, energy_function, dt
                    )
                    logp_valid_prop = energy_function(x_valid_prop)

                    log_accept_ratio = (logp_valid_prop - logp_curr[valid_mask]) + (
                        log_q_backward - log_q_forward
                    )
                    # Sample acceptance (log-space)
                    accept_mask = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
                    accept_valid = accept_mask.float().unsqueeze(-1)

                    # Adaptive step size update
                    print(f"{i}th MCMC current step size: {dt}")
                    acceptance_rate = accept_mask.float().mean().item()
                    if acceptance_rate > 0.55:
                        dt *= 1.1
                    else:
                        dt /= 1.1
                    print(
                        f"{i}th MCMC acceptance rate: {acceptance_rate}, next step size changed to {dt}"
                    )

                    if return_acceptance_rate:
                        acceptance_rate_list.append(acceptance_rate)

                    x_valid = accept_valid * x_valid_prop + (1 - accept_valid) * x_valid
                    accept_mask_float = accept_mask.float()
                    logp_curr[valid_mask] = (
                        accept_mask_float * logp_valid_prop
                        + (1 - accept_mask_float) * logp_curr[valid_mask]
                    )

                    if energy_function.is_molecule:
                        x_valid = remove_mean(
                            x_valid, energy_function.n_particles, energy_function.n_spatial_dim
                        )

                    x_curr = torch.cat([x_valid, x_invalid], dim=0)
            except Exception as e:
                print(f"Error during MALA step: {e}")

        if return_acceptance_rate:
            return x_curr, acceptance_rate_list
        else:
            return x_curr, None
