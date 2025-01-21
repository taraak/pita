from bgflow import GaussianProposal
from src.energies.lennardjones_energy import LennardJonesPotential
import tqdm
from bgflow import  IterativeSampler, SamplerState, MCMCStep
import torch
import numpy as np

dim = 39
n_particles = 13


energy = LennardJonesPotential(dim=dim,
                               n_particles=n_particles,
                               two_event_dims=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_burnin_steps', type=int, default=200000) #200000
    parser.add_argument('--save_file', type=str, default="samples.npy")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--noise_std', type=float, default=0.1)

    args = parser.parse_args()
    temperature = args.temperature
    n_burnin = args.num_burnin_steps
    noise_std = args.noise_std

    sampler_state = SamplerState(samples=[torch.randn(1, n_particles, dim // n_particles)])
    print("Starting sampling at temperature", temperature)

    mcmc_step = MCMCStep(energy,
                     proposal=GaussianProposal(noise_std=noise_std),
                     target_temperatures=temperature)
    
    sampler = IterativeSampler(sampler_state,
                           sampler_steps=[mcmc_step],
                           stride=100,
                           n_burnin=n_burnin,
                           return_hook=lambda samples: [samples[0][:,0]])
    samples = sampler.sample(args.num_samples)
    np.save(args.save_file, samples)
