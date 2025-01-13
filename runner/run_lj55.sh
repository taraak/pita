#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH -J a100_rdem                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

export seed=62;

python src/train.py -m experiment=lj55 trainer=gpu 

