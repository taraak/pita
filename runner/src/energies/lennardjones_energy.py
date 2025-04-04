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
from src.models.components.replay_buffer import ReplayBuffer
from src.utils.data_utils import remove_mean


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

    def _log_prob(self, x, smooth=False, T=1.0):
        E = -self._energy(x, smooth=smooth)
        return (E / self._temperature) * T


class LennardJonesEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        energy_factor=1.0,
        is_molecule=True,
        smooth=False,
        temperature=1.0,
    ):
        self.temperature = temperature
        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        if self.n_particles != 13 and self.n_particles != 55:
            raise NotImplementedError

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.data_path_train = (
            data_path
            + f"LJ{self.n_particles}_temp_{self.temperature:0.1f}/train_split_LJ{self.n_particles}-1000.npy"
        )
        self.data_path_val = (
            data_path
            + f"LJ{self.n_particles}_temp_{self.temperature:0.1f}/val_split_LJ{self.n_particles}-1000.npy"
        )
        self.data_path_test = (
            data_path
            + f"LJ{self.n_particles}_temp_{self.temperature:0.1f}/test_split_LJ{self.n_particles}-1000.npy"
        )

        if self.n_particles == 13:
            self.name = "LJ13_efm"
        elif self.n_particles == 55:
            self.name = "LJ55"

        self.device = device

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

        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def __call__(self, samples: torch.Tensor, T=1.0) -> torch.Tensor:
        return self.lennard_jones._log_prob(samples, smooth=self.smooth, T=T).squeeze(-1)

    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("Data path for training data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.lennard_jones.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_dataset_fig(self, samples, T=1.0, T_og=1.0):
        # import pdb; pdb.set_trace()
        test_data_smaller = self.sample_test_set(5000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        _, bins, _ = axs[0].hist(
            dist_test.view(-1),
            bins=100,
            density=True,
            alpha=0.6,
            histtype="step",
            linewidth=4,
            color="g",
            label="test data",
        )
        axs[0].hist(
            dist_samples.view(-1),
            bins=bins,  # 100,
            alpha=0.6,
            density=True,
            histtype="step",
            color="r",
            linewidth=4,
            label="generated data",
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend()

        energy_samples = -self(samples, T=T).detach().detach().cpu()
        energy_test = -self(test_data_smaller, T=T_og).detach().detach().cpu()

        # min_energy = min(energy_test.min(), energy_samples.min()).item()
        # max_energy = max(energy_test.max(), energy_samples.max()).item()

        if self.n_particles == 13:
            # min_energy = -60
            # max_energy = 60

            min_energy = energy_test.min().item() - 10
            max_energy = energy_test.max().item() + 10

        elif self.n_particles == 55:
            min_energy = -380
            max_energy = -180

        _, bins, _ = axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.6,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=bins,  # 100
            density=True,
            alpha=0.6,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
