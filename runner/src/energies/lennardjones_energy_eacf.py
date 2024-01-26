import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from lightning.pytorch.loggers import WandbLogger

from bgflow import Energy
from bgflow.utils import distance_vectors, distances_from_vectors

from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.replay_buffer import ReplayBuffer
from src.utils.data_utils import remove_mean


def sample_from_array(array, size):
    idx = np.random.choice(array.shape[0], size=size)
    return array[idx]


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential:
    def __init__(
        self,
        dim,
        n_particles,
        eps=1.0,
        rm=1.0,
        tau=1.0,
        harmonic_potential_coef=0.5,
        device="cpu",
    ):
        self._dim = dim
        self._n_particles = n_particles
        self._n_dims = dim // n_particles
        self.eps = eps
        self.rm = rm
        self.tau = tau
        self.harmonic_potential_coef = harmonic_potential_coef
        self.device = device

        self.event_shape = torch.Size([dim])

    def _get_senders_and_receivers_fully_connected(self, n_nodes):
        receivers = []
        senders = []
        for i in range(n_nodes):
            for j in range(n_nodes - 1):
                receivers.append(i)
                senders.append((i + 1 + j) % n_nodes)
        return torch.tensor(senders), torch.tensor(receivers)

    def _energy(self, x: torch.Tensor):
        if isinstance(self.rm, float):
            r = torch.ones(self._n_particles, device=self.device) * self.rm
        senders, receivers = self._get_senders_and_receivers_fully_connected(
            self._n_particles
        )
        vectors = x[senders] - x[receivers]
        d = torch.linalg.norm(vectors, ord=2, dim=-1)
        term_inside_sum = (r[receivers] / d) ** 12 - 2 * (r[receivers] / d) ** 6
        energy = self.eps / (2 * self.tau) * term_inside_sum.sum()

        centre_of_mass = x.mean(dim=0)
        harmonic_potential = (
            self.harmonic_potential_coef * (x - centre_of_mass).pow(2).sum()
        )
        return energy + harmonic_potential

    def _log_prob(self, x: torch.Tensor):
        x = x.reshape(-1, self._n_particles, self._n_dims)
        return -torch.vmap(self._energy)(x)


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
        data_path_train=None,
    ):
        torch.manual_seed(0)  # seed of 0

        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        if self.n_particles != 13:
            raise NotImplementedError

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.device = device

        self.data_path = data_path
        self.data_path_train = data_path_train

        self.lennard_jones = LennardJonesPotential(
            dim=dimensionality,
            n_particles=n_particles,
            eps=1.0,
            rm=1.0,
            tau=1.0,
            harmonic_potential_coef=0.5,
            device=device,
        )

        super().__init__(dimensionality=dimensionality)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.lennard_jones._log_prob(samples)

    def setup_test_set(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        # Following the EACF paper for the partitions
        # This test set is bad. It's a single MC Chain
        test_data = all_data[:1000]
        test_data = remove_mean(test_data, self.n_particles, self.n_spatial_dim)
        test_data = torch.tensor(test_data, device=self.device)
        del all_data
        return test_data
    
    def setup_val_set(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        # Following the EACF paper for the partitions
        # This test set is bad. It's a single MC Chain
        val_data = all_data[1000:2000]
        val_data = remove_mean(
            val_data, self.n_particles, self.n_spatial_dim
        )
        val_data = torch.tensor(val_data,
                                 device=self.device)
        del all_data
        return val_data
    
    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("No train data path provided")
        train_data = np.load(self.data_path_train, allow_pickle=True)
        train_data = remove_mean(train_data, self.n_particles, self.n_spatial_dim)
        train_data = torch.tensor(train_data, device=self.device)
        return train_data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.lennard_jones.event_shape)]
        x = x.view(*batch_shape, self.n_particles, self.n_spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1)
            == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        unprioritized_buffer_samples: Optional[torch.Tensor],
        cfm_samples: Optional[torch.Tensor],
        replay_buffer: ReplayBuffer,
        wandb_logger: WandbLogger,
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

            if unprioritized_buffer_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(
                    f"{prefix}cfm_generated_samples", [cfm_samples_fig]
                )

        self.curr_epoch += 1

    def get_dataset_fig(self, samples):
        test_data_smaller = self.sample_test_set(1000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()

        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = -self(samples).detach().detach().cpu()
        energy_test = -self(test_data_smaller).detach().detach().cpu()

        min_energy = min(energy_test.min(), energy_samples.min()).item()
        max_energy = max(energy_test.max(), energy_samples.max()).item()

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(-30, 0),
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(-30, 0),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        # plt.show()

        fig.canvas.draw()
        return PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
