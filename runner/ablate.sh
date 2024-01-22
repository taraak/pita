#!/usr/bin/env bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=23:59:00
#SBATCH --mem=24G
#SBATCH -c 4

export PYTHONUNBUFFERED=1
micromamba activate active_inference

# for pis_scale in 15 17 20 22 25; do
#     for lr in 5e-3 1e-3 5e-4; do
#         python src/train.py model=pis experiment=gmm energy.should_unnormalize=false model.pis_scale=$pis_scale model.noise_schedule=None trainer.check_val_every_n_epoch=10 model.optimizer.lr=$lr trainer.max_epochs=30 &
#     done
# done
# wait

for pis_scale in 1. 1.5 2.; do
    for lr in 5e-3 1e-3 5e-4; do
        python src/train.py model=pis experiment=many_well energy.should_unnormalize=false model.pis_scale=$pis_scale model.noise_schedule=None trainer.check_val_every_n_epoch=10 model.optimizer.lr=$lr trainer.max_epochs=30 &
    done
done
wait