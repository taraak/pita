import PIL
import torch
import numpy as np

from typing import Optional
from lightning.pytorch.loggers import WandbLogger

import matplotlib.pyplot as plt
from src.utils.data_utils import remove_mean
from bgflow import MultiDoubleWellPotential

from src.energies.base_energy_function import BaseEnergyFunction
from src.models.components.replay_buffer import ReplayBuffer


class MultiDoubleWellEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule = True,
    ):
        torch.manual_seed(0)  # seed of 0
        np.random.seed(0)

        self.n_particles = n_particles
        self.n_spatial_dim = dimensionality // n_particles

        self.curr_epoch = 0
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period

        self.data_normalization_factor = data_normalization_factor

        self.data_path = data_path

        self.device = device

        self.val_set_size = 1000
        self.test_set_size = 1000
        self.train_set_size = 100000

        self.multi_double_well = MultiDoubleWellPotential(
            dim = dimensionality,
            n_particles = n_particles,
            a = 0.9,
            b = -4,
            c = 0,
            offset = 4,
            two_event_dims=False)

        super().__init__(dimensionality=dimensionality,
                         is_molecule=is_molecule)

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return - self.multi_double_well.energy(samples).squeeze(-1)

    def setup_test_set(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        # idx = all_data[1]
        # test_data = all_data[0][idx[-500000:]]
        test_data = all_data[0][-self.test_set_size:]
        test_data = remove_mean(
            test_data, self.n_particles, self.n_spatial_dim
        ).to(self.device)
        del all_data
        return test_data

    def setup_train_set(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        # idx = all_data[1]
        # train_data = all_data[0][idx[:100000]]
        train_data = all_data[0][:self.train_set_size]        
        train_data = remove_mean(
            train_data, self.n_particles, self.n_spatial_dim
        ).to(self.device)
        del all_data
        return train_data
    
    def setup_val_set(self):
        all_data = np.load(self.data_path, allow_pickle=True)
        # idx = all_data[1]
        # val_data = all_data[0][idx[100000:500000]]
        val_data = all_data[0][-self.test_set_size - self.val_set_size : -self.test_set_size]
        val_data = remove_mean(
            val_data, self.n_particles, self.n_spatial_dim
        ).to(self.device)
        del all_data
        return val_data

    def interatomic_dist(self, x):
        batch_shape = x.shape[: -len(self.multi_double_well.event_shape)]
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

        min_energy = -25 
        max_energy = 0
        
        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
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
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
