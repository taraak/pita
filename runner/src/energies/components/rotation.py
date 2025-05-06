import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def create_random_rotation_matrix(batch_size):
    # Generate random numbers from a normal distribution
    u = torch.randn((batch_size, 4))

    # Normalize to get unit quaternions
    norm_u = torch.norm(u, p=2, dim=1, keepdim=True)
    q = u / norm_u

    # Convert quaternions to rotation matrices
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    R[:, 0, 1] = 2 * qx * qy - 2 * qz * qw
    R[:, 0, 2] = 2 * qx * qz + 2 * qy * qw

    R[:, 1, 0] = 2 * qx * qy + 2 * qz * qw
    R[:, 1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    R[:, 1, 2] = 2 * qy * qz - 2 * qx * qw

    R[:, 2, 0] = 2 * qx * qz - 2 * qy * qw
    R[:, 2, 1] = 2 * qy * qz + 2 * qx * qw
    R[:, 2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return R

import torch

def random_rotation_matrices(batch_size, device='cpu', dtype=torch.float32):
    # Step 1: Sample from normal distribution
    A = torch.randn((batch_size, 3, 3), device=device, dtype=dtype)

    # Step 2: QR decomposition
    Q, R = torch.linalg.qr(A)

    # Step 3: Ensure right-handed system (det = +1)
    d = torch.det(Q)
    d = d.sign().unsqueeze(-1).unsqueeze(-1)
    Q = Q * d  # flip sign if det = -1

    return Q


class Random3DRotationTransform(torch.nn.Module):
    def __init__(self, num_particles, dim):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim
        self.batch_size = 1024 * 1024
        self.curr_idx = 0
        self.rotation_matrices = random_rotation_matrices(self.batch_size, dtype=torch.float32)

    def forward(self, data, force=None):
        data = data.reshape(-1, self.num_particles, self.dim)  # batch dimension needed for einsum
        rot = self.get_rot_like(data)
        data = torch.einsum("bij,bki->bkj", rot, data)
        data = data.reshape(-1, self.num_particles * self.dim)  # don't want to return with batch dim

        if force is not None:
            force = force.reshape(-1, self.num_particles, self.dim)
            force = torch.einsum("bij,bki->bkj", rot, force)
            force = force.reshape(-1, self.num_particles * self.dim)
            return data, force
        return data

    def get_rot_like(self, data):
        return torch.tensor(R.random(len(data)).as_matrix()).to(data)


class Random2DRotationTransform(torch.nn.Module):
    def __init__(self, num_particles, dim):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim

    def forward(self, data):
        # rotation augmentation
        data = data.reshape(-1, self.num_particles, self.dim)  # batch dimension needed for einsum
        x = torch.rand(len(data)) * 2 * np.pi
        s = torch.sin(x)
        c = torch.cos(x)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute(2, 0, 1)
        data = torch.einsum("bij,bki->bkj", rot, data)
        data = data.reshape(self.num_particles * self.dim)  # don't want to return with batch dim
        return data


if __name__ == "__main__":
    # Example usage
    batch_size = 5
    rotation_matrices = create_random_rotation_matrix(batch_size)
    print(rotation_matrices.shape)  # Should print (5, 3, 3)
