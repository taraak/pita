import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.rwkernel import RandomWalkKernel
from pyro.ops.integrator import potential_grad

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
    return -lj_energies


class Langevin(RandomWalkKernel):
    def __init__(self, potential_fn, step_size):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def sample(self, params):
        param_grad = potential_grad(self.potential_fn, params)
        step_size = math.exp(self._log_step_size)
        new_params = {
            k: v
            - step_size * param_grad[k]
            + math.sqrt(2 * step_size)
            * torch.randn(v.shape, dtype=v.dtype, device=v.device)
            for k, v in params.items()
        }
        energy_proposal = self.potential_fn(new_params)
        delta_energy = energy_proposal - self._energy_last

        accept_prob = (-delta_energy).exp().clamp(max=1.0).item()
        rand = pyro.sample(
            "rand_t={}".format(self._t),
            dist.Uniform(0.0, 1.0),
        )
        accepted = False
        if rand < accept_prob:
            accepted = True
            params = new_params
            self._energy_last = energy_proposal

        if self._t <= self._warmup_steps:
            adaptation_speed = max(0.001, 0.1 / math.sqrt(1 + self._t))
            self._log_step_size += adaptation_speed * (
                accept_prob - self.target_accept_prob
            )

        self._t += 1

        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t

        self._mean_accept_prob += (accept_prob - self._mean_accept_prob) / n

        return params.copy()


def model():
    positions = pyro.sample(
        "positions",
        dist.Uniform(-1e2, 1e2).expand([n_particles * 3]).to_event(1),
    )
    potential_energy = energy2(positions).sum() / temperature
    pyro.factor("potential_energy", potential_energy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--save_file", type=str, default="samples.npy")
    parser.add_argument("--init_step_size", type=float, default=0.01)
    parser.add_argument("--target_accept_prob", type=float, default=0.234)
    parser.add_argument("--kernel", type=str, default="nuts")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--from_file",
        type=str,
        default=None,
        #default="/home/mila/a/alexander.tong/feynman-kac-diffusion/data/train_split_LJ13-1000.npy",
    )
    parser.add_argument("--i", type=int, default=0)
    args = parser.parse_args()
    temperature = args.temperature
    if args.kernel == "nuts":
        kernel = NUTS(model)
    elif args.kernel == "langevin":
        kernel = Langevin(model, step_size=args.init_step_size)
    elif args.kernel == "randomwalk":
        kernel = RandomWalkKernel(
            model,
            init_step_size=args.init_step_size,
            target_accept_prob=args.target_accept_prob,
        )
    if args.from_file:
        data = np.load(args.from_file)
        data = data[args.i:args.i+1]
        mcmc = MCMC(
            kernel,
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            initial_params={"positions": torch.tensor(data)},
            num_chains=data.shape[0],
        )
    else:
        mcmc = MCMC(
            kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps
        )
    mcmc.run()
    samples = mcmc.get_samples()
    print(samples)
    np.save(args.save_file, samples["positions"])
