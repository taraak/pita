#!/usr/bin/env bash
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=47:59:00
#SBATCH --mem=32G
#SBATCH -c 6

for seed in {0..4}; do
    # python src/train.py model=pis experiment=gmm energy.should_unnormalize=false model.pis_scale=25. model.noise_schedule=None trainer.check_val_every_n_epoch=10 model.optimizer.lr=5e-4 trainer.max_epochs=100 seed=$seed &
    python src/train.py model=pis experiment=many_well energy.should_unnormalize=false model.pis_scale=2. model.noise_schedule=None trainer.check_val_every_n_epoch=10 model.optimizer.lr=5e-3 trainer.max_epochs=100 seed=$seed &
done
wait