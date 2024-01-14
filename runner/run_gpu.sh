#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
module --quiet load miniconda/3
conda activate actinfenv


export seed=62;

python src/train.py experiment=lj13 trainer=gpu
