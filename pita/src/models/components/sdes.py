import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from src.models.components.utils import (
    compiled_divergence_fn,
    compute_divergence_exact,
    compute_laplacian_exact,
)


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)


@dataclass
class SDETerms:
    drift_X: torch.Tensor
    drift_A: torch.Tensor
    divergence_score: Optional[torch.Tensor] = None
    cross_term: Optional[torch.Tensor] = None
    dUt_dt: Optional[torch.Tensor] = None
    diffusion: Optional[torch.Tensor] = None

    @staticmethod
    def cpu(data):
        """Moves all tensors in the SDETerms instance to CPU."""
        return SDETerms(
            drift_X=data.drift_X.cpu(),
            drift_A=data.drift_A.cpu(),
            divergence_score=data.divergence_score.cpu()
            if data.divergence_score is not None
            else None,
            cross_term=data.cross_term.cpu() if data.cross_term is not None else None,
            dUt_dt=data.dUt_dt.cpu() if data.dUt_dt is not None else None,
            diffusion=data.diffusion.cpu() if data.diffusion is not None else None,
        )

    @staticmethod
    def concatenate(data_list):
        """Concatenates corresponding tensors from a list of DataClass instances."""
        if not data_list:
            raise ValueError("The data_list is empty.")

        drift_X_cat = torch.cat([data.drift_X for data in data_list], dim=0)
        drift_A_cat = torch.cat([data.drift_A for data in data_list], dim=0)
        divergence_score_cat = (
            torch.cat([data.divergence_score for data in data_list], dim=0)
            if data_list[0].divergence_score is not None
            else None
        )
        cross_term_cat = (
            torch.cat([data.cross_term for data in data_list], dim=0)
            if data_list[0].cross_term is not None
            else None
        )
        dUt_dt_cat = (
            torch.cat([data.dUt_dt for data in data_list], dim=0)
            if data_list[0].dUt_dt is not None
            else None
        )
        diffusion_cat = (
            torch.cat([data.diffusion for data in data_list], dim=0)
            if data_list[0].diffusion is not None
            else None
        )
        return SDETerms(
            drift_X=drift_X_cat,
            drift_A=drift_A_cat,
            divergence_score=divergence_score_cat,
            cross_term=cross_term_cat,
            dUt_dt=dUt_dt_cat,
            diffusion=diffusion_cat,
        )


class VEReverseSDE:
    def __init__(
        self,
        noise_schedule,
        energy_net=None,
        score_net=None,
        cdf=None,
        pin_energy=False,
        debias_inference=True,
    ):
        super().__init__()
        self.energy_net = energy_net
        self.score_net = score_net
        self.noise_schedule = noise_schedule
        self.pin_energy = pin_energy
        self.debias_inference = debias_inference
        self.compiled_divergence_fn = cdf
        if self.compiled_divergence_fn is None:
            self.compiled_divergence_fn = compiled_divergence_fn(self.score_net.forward)
        # self.compiled_divergence_fn = compiled_divergence_fn(self.score_net.forward)
        self.is_compiled = False

    def f_not_debiased(self, t, x, beta, gamma_energy):
        assert self.score_net is not None
        ht = self.noise_schedule.h(t)
        drift_X = (
            gamma_energy * (self.score_net(ht, x, beta) * self.g(t).pow(2).unsqueeze(-1)).detach()
        )
        drift_A = torch.zeros(x.shape[0]).to(x.device).detach()
        sde_terms = SDETerms(
            drift_X=drift_X,
            drift_A=drift_A,
        )
        return sde_terms

    def f(
        self,
        t,
        x,
        beta,
        gamma_energy_schedule,
        gamma_score,
        energy_function,
        resampling_interval=-1,
    ):
        gamma_energy = gamma_energy_schedule.gamma(t)

        # TODO: we are making them equal here should change code
        gamma_score = gamma_energy

        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)
        if not self.debias_inference:
            return self.f_not_debiased(t, x, beta, gamma_energy)

        assert self.energy_net is not None

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)

            ht = self.noise_schedule.h(t)
            nabla_Ut = self.energy_net(
                ht, x, beta, pin=self.pin_energy, t=t, energy_function=energy_function
            )

            if self.score_net is not None:
                s_t = self.score_net(
                    ht,
                    x,
                    beta,
                )
                bt = s_t * self.g(t).pow(2).unsqueeze(-1) / 2
            else:
                bt = -nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2

            drift_X = (
                gamma_energy * -nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2 + gamma_score * bt
            ).detach()

            drift_A = torch.zeros(x.shape[0]).to(x.device).detach()

            if resampling_interval == -1:
                sde_terms = SDETerms(
                    drift_X=drift_X,
                    drift_A=drift_A,
                )

            Ut = self.energy_net.forward_energy(
                ht=ht,
                xt=x,
                beta=beta,
                pin=self.pin_energy,
                energy_function=energy_function,
                t=t,
            )

            if self.score_net is not None:
                if not self.is_compiled and self.trainer.num_nodes > 1:
                    for i in range(self.trainer.world_size):
                        self.trainer.strategy.barrier()
                        print(f"Rank {i} computing divergence")
                        if i == self.trainer.strategy.node_rank:
                            div_st = self.compiled_divergence_fn(ht, x, beta).detach()
                else:
                    div_st = self.compiled_divergence_fn(ht, x, beta).detach()
                self.is_compiled = True
                div_bt = div_st * self.g(t).pow(2) / 2
            else:
                laplacian_Ut = compute_laplacian_exact(
                    partial(
                        self.energy_net.forward_energy,
                        pin=self.pin_energy,
                        energy_function=energy_function,
                        t=t,
                    ),
                    ht,
                    x,
                    beta,
                ).detach()
                div_bt = -laplacian_Ut * (self.g(t).pow(2) / 2)

            dUt_dt = torch.autograd.grad(Ut.sum(), t, create_graph=True)[0].detach()

            inner_prod = (-nabla_Ut * bt).sum(-1).detach()

            drift_A = (
                gamma_energy * gamma_score * inner_prod
                + gamma_score * div_bt
                + gamma_energy * dUt_dt
                + gamma_energy_schedule.dgamma_dt(t) * Ut
            )
            # clip the drift
            # drift_A = torch.clamp(drift_A - drift_A.mean() , max=10)
            drift_A = torch.clamp(drift_A, max=torch.quantile(drift_A, 0.9))

        sde_terms = SDETerms(
            drift_X=drift_X,
            drift_A=drift_A,
            divergence_score=div_bt,
            cross_term=inner_prod,
            dUt_dt=dUt_dt,
        )
        return sde_terms

    def g(self, t):
        g = self.noise_schedule.g(t)
        return g

    def diffusion(self, t, x, diffusion_scale):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        diffusion = diffusion_scale * self.g(t)[:, None] * torch.randn_like(x).to(x.device)
        return diffusion
