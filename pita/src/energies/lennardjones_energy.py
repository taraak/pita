from contextlib import contextmanager
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors
from hydra.utils import get_original_cwd
from lightning.pytorch.loggers import WandbLogger
from scipy.interpolate import CubicSpline
from src.energies.base_energy_function import BaseEnergyFunction
from src.energies.base_molecule_energy_function import BaseMoleculeEnergy
from src.models.components.replay_buffer import ReplayBuffer
from src.utils.data_utils import remove_mean


@contextmanager
def conditional_grad(condition):
    if condition:
        with torch.enable_grad():
            yield
    else:
        yield


def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    # p=0.9
    # filter = (r < p)
    # lj = lj * ~filter + filter * (energy_slope(p) * (r-p) +  ((1 / p) ** 12 - 2 * (1 / p) ** 6))
    return lj


def cubic_spline(x_new, x, c):
    x, c = x.to(x_new.device), c.to(x_new.device)
    intervals = torch.bucketize(x_new, x) - 1
    intervals = torch.clamp(intervals, 0, len(x) - 2)  # Ensure valid intervals
    # Calculate the difference from the left breakpoint of the interval
    dx = x_new - x[intervals]
    # Evaluate the cubic spline at x new
    y_new = (
        c[0, intervals] * dx**3
        + c[1, intervals] * dx**2
        + c[2, intervals] * dx
        + c[3, intervals]
    )
    return y_new


class LennardJonesPotential(Energy):
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        two_event_dims=True,
        energy_factor=1.0,
        range_min=0.65,
        range_max=2.0,
        interpolation=1000,
        temperature=1.0,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor

        self._temperature = temperature

        self.range_min = range_min
        self.range_max = range_max

        # fit spline cubic on these ranges
        interpolate_points = torch.linspace(range_min, range_max, interpolation)
        es = lennard_jones_energy_torch(interpolate_points, self._eps, self._rm)
        coeffs = CubicSpline(np.array(interpolate_points), np.array(es)).c
        self.splines = partial(cubic_spline, x=interpolate_points, c=torch.tensor(coeffs).float())

    def _energy(self, x, smooth=False):
        batch_shape = x.shape[: -len(self.event_shape)]
        x = x.view(*batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)

        if smooth:
            filter = dists < self.range_min
            lj_energies = lj_energies * ~filter + filter * self.splines(dists).squeeze(-1)

            # lj_energies[dists < self.range_min] = self.splines(dists[dists < self.range_min]).squeeze(-1)
            # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.view(*batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(*batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale

        return lj_energies[:, None]

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        return x - torch.mean(x, dim=1, keepdim=True)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy(x).cpu().numpy()

    def _log_prob(self, x, smooth=False):
        E = -self._energy(x, smooth=smooth)
        return E / self._temperature


class LennardJonesEnergy(BaseMoleculeEnergy):
    def __init__(
        self,
        dimensionality,
        n_particles,
        spatial_dim,
        data_path,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        energy_factor=1.0,
        is_molecule=True,
        smooth=False,
        temperature=1.0,
        should_normalize=False,
        data_normalization_factor=1.0,
        *args,
        **kwargs,
    ):
        if n_particles != 13 and n_particles != 55:
            raise NotImplementedError
        if n_particles == 13:
            self.name = "LJ13_efm"
        elif n_particles == 55:
            self.name = "LJ55"

        super().__init__(
            dimensionality=dimensionality,
            n_particles=n_particles,
            spatial_dim=spatial_dim,
            data_path=data_path,
            data_name="LJ",
            device=device,
            temperature=temperature,
            should_normalize=should_normalize,
            data_normalization_factor=data_normalization_factor,
            plot_samples_epoch_period=plot_samples_epoch_period,
            plotting_buffer_sample_size=plotting_buffer_sample_size,
            is_molecule=is_molecule,
        )

        self.lennard_jones = LennardJonesPotential(
            dim=dimensionality,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            oscillator=True,
            oscillator_scale=1.0,
            two_event_dims=False,
            energy_factor=energy_factor,
            temperature=self.temperature,
        )

        self.smooth = smooth

    def __call__(self, samples: torch.Tensor, return_force=False) -> torch.Tensor:
        with conditional_grad(return_force):
            samples_requires_grad = samples.requires_grad
            samples.requires_grad = True

            if self.should_normalize:
                samples = self.unnormalize(samples)

            logprobs = self.lennard_jones._log_prob(samples, smooth=self.smooth).squeeze(-1)
            if return_force:
                forces = torch.autograd.grad(logprobs.sum(), samples, create_graph=False)[0]
                samples.requires_grad = samples_requires_grad
                return logprobs.detach(), forces.detach()

            return logprobs.detach()
