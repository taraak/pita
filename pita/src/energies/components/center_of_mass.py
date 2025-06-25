from typing import Any

import numpy as np
import torch


class CenterOfMassTransform(torch.nn.Module):
    """Applies Gaussian noise to the center of mass of the molecule."""

    def __init__(self, n_particles: int, n_spatial_dim: int) -> None:
        """
        Args:
            std (float): Standard deviation of the Gaussian noise to be added.
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()
        self.std = 1 / np.sqrt(n_particles * n_spatial_dim)
        self.num_particles = n_particles
        self.num_dimensions = n_spatial_dim

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # batch x n_particles x n_spatial_dim
        data = data.reshape(-1, self.num_particles, self.num_dimensions)

        # Generate noise and adjust the center of mass
        noise = torch.randn((data.shape[0], 1, self.num_dimensions), device=data.device) * self.std

        # Shift all particles so that the center of mass is moved
        data = data + noise

        return data.reshape(-1, self.num_particles * self.num_dimensions)
