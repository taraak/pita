from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
from src.models.components.temperature_schedules import ConstantInvTempSchedule
from src.models.components.utils import (
    compute_divergence_exact,
    compute_divergence_forloop,
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


class VEReverseSDE(torch.nn.Module):
    def __init__(
        self, noise_schedule, energy_net=None, score_net=None, pin_energy=False, debias_inference=True
    ):
        super().__init__()
        self.energy_net = energy_net
        self.score_net = score_net
        self.noise_schedule = noise_schedule
        self.pin_energy = pin_energy
        self.debias_inference = debias_inference

    def f_not_debiased(self, t, x, beta):
        assert self.score_net is not None
        h_t = self.noise_schedule.h(t)
        drift_X = self.score_net(h_t, x, beta) * self.g(t).pow(2).unsqueeze(-1)
        drift_A = torch.zeros(x.shape[0]).to(x.device)
        sde_terms = SDETerms(
            drift_X=drift_X,
            drift_A=drift_A,
        )
        return sde_terms

    def f(self, t, x, beta, gamma, energy_function, resampling_interval=-1):
        assert self.energy_net is not None
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        if not self.debias_inference:
            return self.f_not_debiased(t, x, beta)

        with torch.enable_grad():
            x.requires_grad_(True)
            t.requires_grad_(True)

            h_t = self.noise_schedule.h(t)
            nabla_Ut = self.energy_net(
                h_t, x, beta, pin=self.pin_energy, t=t, energy_function=energy_function
            )

            if self.score_net is not None:
                s_t = self.score_net(
                    h_t,
                    x,
                    beta,
                )
                bt = s_t * self.g(t).pow(2).unsqueeze(-1) / 2
            else:
                bt = -nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2

            drift_X = (gamma * (-nabla_Ut * self.g(t).pow(2).unsqueeze(-1) / 2 + bt)).detach()

            drift_A = torch.zeros(x.shape[0]).to(x.device).detach()

            if resampling_interval == -1:
                sde_terms = SDETerms(
                    drift_X=drift_X,
                    drift_A=drift_A,
                )

            Ut = self.energy_net.forward_energy(
                h_t=h_t,
                x_t=x,
                beta=beta,
                pin=self.pin_energy,
                energy_function=energy_function,
                t=t,
            )

            if self.score_net is not None:
                div_st = compute_divergence_exact(
                    self.score_net.forward,
                    h_t,
                    x,
                    beta,
                ).detach()
                div_bt = div_st * self.g(t).pow(2) / 2
            else:
                laplacian_Ut = compute_laplacian_exact(
                    partial(
                        self.energy_net.forward_energy,
                        pin=self.pin_energy,
                        energy_function=energy_function,
                        t=t,
                    ),
                    h_t,
                    x,
                    beta,
                ).detach()
                div_bt = -laplacian_Ut * (self.g(t).pow(2) / 2)

            dUt_dt = torch.autograd.grad(Ut.sum(), t, create_graph=True)[0].detach()

            inner_prod = (-nabla_Ut * bt).sum(-1).detach()

            # print("dUt_dt", dUt_dt[:2])
            # print("div_st", div_bt[:2])
            # print("inner_prod", inner_prod[:2])

            drift_A = gamma**2 * inner_prod + gamma * div_bt + gamma * dUt_dt

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
