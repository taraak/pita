#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

export seed=62;

python src/train.py experiment=dw4 trainer=gpu #model.resampling_interval=null

