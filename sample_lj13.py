import numpy as np
import pyro
from pyro.infer import MCMC, NUTS
import pyro.distributions as dist
import torch

n_particles = 13


# Define the Lennard-Jones 13 potential function
def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


def remove_mean(x):
    return x - torch.mean(x, dim=1, keepdim=True)


def energy2(x):
    x = x.reshape(-1, n_particles, 3)
    dists = torch.vmap(torch.pdist)(x)
    lj_energies = lennard_jones_energy_torch(dists).sum(dim=(-1))
    osc_energies = 0.5 * remove_mean(x).pow(2).sum(dim=(-2, -1))
    lj_energies = 2 * lj_energies + osc_energies
    return lj_energies


def model():
    positions = pyro.sample(
        "positions",
        dist.Normal(torch.zeros(n_particles * 3), torch.ones(n_particles * 3)),
    )
    potential_energy = energy2(positions).sum() / temperature
    pyro.factor("potential_energy", -potential_energy)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--save_file', type=str, default="samples.npy")
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    temperature = args.temperature
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps)
    mcmc.run()
    samples = mcmc.get_samples()
    np.save(args.save_file, samples["positions"])
