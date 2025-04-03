from contextlib import contextmanager

import numpy as np
import torch
from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.sdes import SDETerms, VEReverseSDE
from src.models.components.utils import sample_cat_sys
from src.utils.data_utils import remove_mean
from lightning import LightningModule
from dataclasses import dataclass, asdict, fields
from tqdm.auto import tqdm
from rich.progress import Progress
from pytorch_lightning.utilities.rank_zero import rank_zero_only

disable = rank_zero_only.rank != 0  

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


class WeightedSDEIntegrator:
    def __init__(self,
            sde: VEReverseSDE,
            num_integration_steps: int,
            start_resampling_step: int,
            end_resampling_step: int,
            lightning_module: LightningModule, 
            reverse_time: bool = True,
            diffusion_scale=1.0,
            time_range=1.0,
            resampling_interval=-1,
            num_negative_time_steps=100,
            batch_size=None,
            no_grad=True,
            resample_at_end=False,
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
        self.batch_size = batch_size
        self.no_grad = no_grad
        self.resample_at_end = resample_at_end
        self.lightning_module = lightning_module

        self.start_time = time_range if reverse_time else 0.0
        self.end_time = time_range - self.start_time



    def integrate_sde(self,
        x1: torch.Tensor,
        energy_function: BaseEnergyFunction,
        inverse_temperature=1.0,
        annealing_factor=1.0,
        resampling_interval=None
    ):
        if self.batch_size is None:
            self.batch_size = x1.shape[0]

        if resampling_interval is None:
            resampling_interval = self.resampling_interval

        print("The resampling interval is: ", resampling_interval)


        times = torch.linspace(
            self.start_time, self.end_time, self.num_integration_steps + 1, device=x1.device
        )[:-1]

        dt = self.time_range / self.num_integration_steps

        x = x1
        x1.requires_grad = True
        samples = []
        logweights = []
        num_unique_idxs = []
        sde_terms_all = []
        a = torch.zeros(x.shape[0], device=x.device)
        # with Progress() as pb:
        #     t1 = pb.add_task('inner', total=len(times))
        #     # t2 = pb.add_task('outer', total=100)

        with conditional_no_grad(self.no_grad):
            # change for loop to tqdm for progress bar
            # with tqdm(total=len(times), position=0, leave=True) as pbar:
            # for i in tqdm((foo_, range_ ), position=0, leave=True):
            #     pbar.update()

            for step, t in enumerate(tqdm(times, position=0, leave=True, dynamic_ncols=True, disable=disable)):
            #for step, t in enumerate(times):
                x, a, idxs, sde_terms = self.ddp_batched_euler_maruyama_step(
                    t,
                    x,
                    a,
                    dt,
                    step,
                    inverse_temperature=inverse_temperature,
                    annealing_factor=annealing_factor,
                    energy_function=energy_function,
                    resampling_interval=resampling_interval,
                )
                if energy_function.is_molecule:
                    x = remove_mean(
                        x, energy_function.n_particles, energy_function.n_spatial_dim
                    )


                samples.append(x)
                logweights.append(a)
                num_unique_idxs.append(idxs)
                sde_terms_all.append(sde_terms)

            if self.resample_at_end:
                t = torch.tensor(self.end_time).to(x.device)
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
                logq_0 = -model_energy
                a_next = target_logprob - logq_0 + a
                choice, _ = sample_cat_sys(x.shape[0], a_next)
                x = x[choice]
                logweights.append(a_next)
                samples.append(x)
                num_unique_idxs.append(len(np.unique(choice)))

        samples = torch.stack(samples)
        logweights = torch.stack(logweights)

        if self.num_negative_time_steps > 0:
            print("doing negative time descent...")
            samples_langevin = negative_time_descent(
                x,
                energy_function,
                num_steps=self.num_negative_time_steps,
            )
            samples = torch.concatenate((samples, samples_langevin), axis=0)
        return samples, logweights, num_unique_idxs, sde_terms_all
    

    def ddp_batched_euler_maruyama_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        a: torch.tensor,
        dt: float,
        step: int,
        inverse_temperature: float,
        annealing_factor: float,
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
            energy_function,
            resampling_interval,
        )

        # gather x and a
        x_next = self.lightning_module.all_gather(x_next).reshape(-1, *x_next.shape[1:])
        a_next = self.lightning_module.all_gather(a_next).reshape(-1, *a_next.shape[1:])

        sde_terms = self.lightning_module.all_gather(asdict(sde_terms))
        for term in fields(SDETerms):
            sde_terms[term.name] = sde_terms[term.name].reshape(-1, *sde_terms[term.name].shape[2:])
        sde_terms = SDETerms(**sde_terms)

        # don't start accumulating weights until step start_resamplings_step
        if step < self.start_resampling_step:
            a_next = torch.zeros_like(a_next)
            x_next = x  # samples are distributed according to the prior, don't move
        elif step >= self.end_resampling_step:
            a_next = torch.zeros_like(a_next)

        if (
            resampling_interval == -1
            or (step + 1) % resampling_interval != 0
            or step < self.start_resampling_step
        ):
            return x_next, a_next, len(x_next), sde_terms

        # resample based on the weights
        choice, _ = sample_cat_sys(x_next.shape[0], a_next)
        x_next = x_next[choice]
        a_next = torch.zeros_like(a_next)

        num_unique_idxs = len(np.unique(choice))

        return x_next, a_next, num_unique_idxs, sde_terms
        

    def euler_maruyama_step(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            a: torch.tensor,
            dt: float,
            inverse_temperature: float,
            annealing_factor: float,
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
                gamma=annealing_factor,
                energy_function=energy_function,
            )
            sde_term.diffusion = self.sde.diffusion(
                t, x[i * self.batch_size : (i + 1) * self.batch_size], self.diffusion_scale
            )
            sde_terms.append(sde_term)

        if x.shape[0] % self.batch_size != 0:
            i = x.shape[0] // self.batch_size
            sde_term = self.sde.f(
                t,
                x[i * self.batch_size :],
                resampling_interval=resampling_interval,
                beta=inverse_temperature,
                gamma=annealing_factor,
                energy_function=energy_function,
            )
            sde_term.diffusion = self.sde.diffusion(t, x[i * self.batch_size :], self.diffusion_scale)
            sde_terms.append(sde_term)
        sde_terms = SDETerms.concatenate(sde_terms)

        # update x, log weights, and log density
        dx = sde_terms.drift_X * dt + sde_terms.diffusion * np.sqrt(dt)
        x_next = x + dx
        a_next = a + sde_terms.drift_A * dt

        return x_next, a_next, sde_terms