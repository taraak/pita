#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
module --quiet load miniconda/3
conda activate actinfenv


export seed=62;



#python src/train.py experiment=lj55 trainer=gpu model.debug_use_train_data=True
python src/train.py experiment=dw4 trainer=gpu model.debug_use_train_data=True
#python src/train.py experiment=lj13 trainer=gpu model.cfm_prior_std=3 model.use_otcfm=True
#python src/train.py experiment=gmm trainer=gpu model.logz_with_cfm=True

