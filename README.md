<div align="center">

# Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts

[![Preprint](http://img.shields.io/badge/paper-arxiv.2402.06121-B31B1B.svg)](https://arxiv.org/pdf/2503.02819)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description

This is the repository for the temperature annealing experiment of [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://arxiv.org/pdf/2503.02819).
It is built on the repository of [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://github.com/jarridrb/DEM/tree/main)

For this experiment, we propose a weighted simulation scheme based on the celebrated Feynman-Kac formula for amortized sampling of Boltzmann distributions using iDEM, 
via inference-time temperature annealing.

We experiment on a 2D GMM task, using the ground truth score in [this notebook](https://github.com/necludov/feynman-kac-diffusion/blob/boltzmann_sampling_public/runner/notebooks/gmm_temp_annealed_birth_death.ipynb) as well as the LJ13 -- the 13-particle Lennard-Jones potential (39 dimensions total).
This code was taken from an internal repository and as such all commit history is lost here.

## Installation

For installation, we recommend the use of Micromamba. Please refer [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for an installation guide for Micromamba.
First, we install dependencies

```bash
# clone project
git clone git@github.com:jarridrb/DEM.git
cd DEM

# create micromamba environment
micromamba create -f environment.yaml
micromamba activate dem

# install requirements
pip install -r requirements.txt

```

Note that the hydra configs interpolate using some environment variables set in the file `.env`. We provide
an example `.env.example` file for convenience. Note that to use wandb we require that you set WANDB_ENTITY in your
`.env` file.

We have also provided a checkpoint for the pre-trained DEM model tranined at temperature 2.0 and used in the experiments in the paper [here](https://github.com/necludov/feynman-kac-diffusion/blob/boltzmann_sampling_public/runner/pretrained_dem_ckpts/epoch_179.ckpt).

To run an experiment, e.g., LJ-13 inference, with temperature annealed to 1.5 you can run on the command line

```bash
python runner/src/eval.py ckpt_path=runner/pretrained_dem_ckpts/epoch_179.ckpt experiment=lj13_tempdem_1.5
```

We include configs for annealing to temperature 1.5 as an example (point to config). 
- the Feynman-Kac Correctors can be turned on and off by changing the **resampling_interval** parameter 
(-1 indicates no resampling, 1 indicates resampling at every step which is what is used for the results in the paper).
- for the **tempered noise** and **target score SDEs** the **scale_diffusion** parameter has
to be set to 1 and -1 respectively.
- the **start_resampling_step** indicates at what integration step to start using FKC. This value is adjusted based on the noise schedule and the variance of 
the weights. We recommend values between 10 to 30 for the temperatures we tried. 

## Citations

If this codebase is useful towards other research efforts please consider citing us.

```
@misc{skreta2025feynmankac,
    title={Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts},
    author={Marta Skreta and Tara Akhound-Sadegh and Viktor Ohanesian and Roberto Bondesan and Alán Aspuru-Guzik and Arnaud Doucet and Rob Brekelmans and Alexander Tong and Kirill Neklyudov},
    year={2025},
    eprint={2503.02819},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Contribute

We welcome issues and pull requests (especially bug fixes) and contributions.
We will try our best to improve readability and answer questions!

## Licences

This repo is licensed under the [MIT License](https://opensource.org/license/mit/).

