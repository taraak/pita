<div align="center">

# Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities

[![Preprint](http://img.shields.io/badge/paper-arxiv.2506.16471-B31B1B.svg)](https://www.arxiv.org/abs/2506.16471)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description

This is the repository for the [Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](https://www.arxiv.org/abs/2506.16471).
It is partially built on the repository of [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://github.com/jarridrb/DEM/tree/main)

In this paper, we propose to train a series of energy-based models using diffusion, to iteratively sample from a series of temperature annealed Boltzmann distributions. We first train at an easy-to-sample high-temperature target first, using data collected from Molecular Dynamics (MD). Then we progressively simulate at lower temperatures until reaching the desired target temperature (using ideas from [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://arxiv.org/abs/2503.02819).

We experiment on a Lennard-Jones system of 13 particles (LJ-13), as well as small peptide molecules (Alanine Dipeptide and Tripeptide).

## Installation

For installation, we recommend the use of Micromamba. Please refer [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for an installation guide for Micromamba.
First, we install dependencies

```bash
# clone project
git clone git@github.com:taraak/pita.git
cd temperature_annealing/pita

# create micromamba environment
micromamba create -f environment.yaml
micromamba activate pita

```

Note that the hydra configs interpolate using some environment variables set in the file `.env`. We provide
an example `.env.example` file for convenience. Note that to use wandb we require that you set WANDB_ENTITY in your
`.env` file.

## Datasets
You can download the datasets for all the experiments from [here](https://data.mendeley.com/datasets/jnrdksfsyp/1) and store them in `data` folder in the main directory. 

## Citations

If this codebase is useful towards other research efforts please consider citing us.

```
@misc{akhoundsadegh2025progressiveinferencetimeannealingdiffusion,
      title={Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities},
      author={Tara Akhound-Sadegh and Jungyoon Lee and Avishek Joey Bose and Valentin De Bortoli and Arnaud Doucet and Michael M. Bronstein and Dominique Beaini and Siamak Ravanbakhsh and Kirill Neklyudov and Alexander Tong},
      year={2025},
      eprint={2506.16471},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.16471},
}
```

## Contribute

We welcome issues and pull requests (especially bug fixes), and contributions.
We will try our best to improve readability and answer questions!

## Licenses

This repo is licensed under the [MIT License](https://opensource.org/license/mit/).
