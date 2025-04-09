import numpy as np
import torch

class PermutationTransform(torch.nn.Module):
    def __init__(self, num_particles, dim):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim

    def forward(self, data):
        assert len(data.shape) == 1, "only process single molecules"
        data = data.reshape(self.num_particles, self.dim)
        # Generate a random permutation of particle indices
        perm = torch.randperm(self.num_particles)
        # Apply the permutation
        data = data[perm]
        # Reshape back to original shape
        data = data.reshape(self.num_particles * self.dim)
        return data

if __name__ == "__main__":
    pass
