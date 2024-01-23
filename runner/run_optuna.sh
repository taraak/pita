#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
module --quiet load miniconda/3
conda activate actinfenv


export seed=62;

#python src/train.py experiment=dw4 trainer=gpu hparams_search=lj13_optuna launcher=mila_cluster
python src/train.py experiment=lj13 trainer=gpu hparams_search=lj13_optuna launcher=mila_cluster
#python src/train.py experiment=many_well trainer=gpu hparams_search=many_well_optuna launcher=mila_cluster

#SBATCH --gres=gpu:1

