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


class BaseMoleculeEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality: int,
        n_particles: int,
        spatial_dim: int,
        data_path: str,
        data_name: str, 
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        is_molecule=True,
        temperature=1.0,
        should_normalize=False,
        data_normalization_factor=1.0,
    ):
                
        self.temperature = temperature
        self.n_particles = n_particles
        self.n_spatial_dim = spatial_dim
        assert self.n_spatial_dim * self.n_particles == dimensionality

        self.device = device

        self.should_normalize = should_normalize
        self.data_normalization_factor = data_normalization_factor

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_path_train = (
            data_path
            + f"{data_name}{self.n_particles}_temp_{self.temperature:0.1f}/train_split_{data_name}{self.n_particles}-10000.npy"
        )
        self.data_path_val = (
            data_path
            + f"{data_name}{self.n_particles}_temp_{self.temperature:0.1f}/val_split_{data_name}{self.n_particles}-10000.npy"
        )
        self.data_path_test = (
            data_path
            + f"{data_name}{self.n_particles}_temp_{self.temperature:0.1f}/test_split_{data_name}{self.n_particles}-10000.npy"
        )

        super().__init__(dimensionality=dimensionality,
                         is_molecule=is_molecule)


    def setup_test_set(self):
        data = np.load(self.data_path_test, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_train_set(self):
        if self.data_path_train is None:
            raise ValueError("Data path for training data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        if self.should_normalize:
            data = self.normalize(data)
        else:
            data = remove_mean(data, self.n_particles, self.n_spatial_dim)
        data = torch.tensor(data, device=self.device)
        return data

    def interatomic_dist(self, x):
        if self.should_normalize:
            x = self.unnormalize(x)
        batch_shape = x.shape[:-1]
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
        latest_samples_not_resampled: Optional[torch.Tensor]=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples,
                                               energy_samples=latest_energies,
                                               samples_not_resampled=latest_samples_not_resampled)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

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

    def get_dataset_fig(self, samples, energy_samples=None, samples_not_resampled=None):
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
        if samples_not_resampled is not None:
            dist_samples_not_resampled = self.interatomic_dist(samples_not_resampled).detach().cpu()
            axs[0].hist(
                dist_samples_not_resampled.view(-1),
                bins=bins,
                alpha=0.6,
                density=True,
                histtype="step",
                color="b",
                linewidth=4,
                label="generated data (not resampled)",
            )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend()

        if energy_samples is None:
            energy_samples = self(samples)
        energy_samples = - energy_samples.detach().cpu()
        energy_test = -self(test_data_smaller).detach().cpu()


        min_energy = energy_test.min().item() - 10
        max_energy = energy_test.max().item() + 10


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

        if samples_not_resampled is not None:
            energy_samples_not_resampled = -self(samples_not_resampled).detach().cpu()
            axs[1].hist(
                energy_samples_not_resampled,
                bins=bins,  # 100
                density=True,
                alpha=0.6,
                range=(min_energy, max_energy),
                color="b",
                histtype="step",
                linewidth=4,
                label="generated data (not resampled)",
            )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
