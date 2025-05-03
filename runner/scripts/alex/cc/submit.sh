#!/bin/bash
#SBATCH --mem-per-cpu=24G
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-3  # Adjust the range to match number of sigma_min values

# Define different sigma_min values
SIGMA_MIN_LIST=(0.002 0.01 0.05 0.1)
SIGMA_MIN=${SIGMA_MIN_LIST[$SLURM_ARRAY_TASK_ID]}

module purge
module load python/3.11 cuda/12.2
module load openmm/8.2.0
module load arrow
module load httpproxy/1.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r cc_requirements_alex_v2.txt
cd runner/
pip install -e .

HYDRA_FULL_ERROR=1 python -u src/train.py \
+trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP","dit","sigma_sweep_v2"] \
model/net=dit \
model.noise_schedule.sigma_min=0.01 \
model.dem.num_training_epochs=0 \
trainer.check_val_every_n_epoch=50 \
model.dem.num_training_epochs=0 \
model.debias_inference=True \
model.resample_at_end=False \
model.loss_weights.energy_matching=1.0 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.0 \
model.inference_batch_size=1024 \
model.num_negative_time_steps=100 \
model.end_resampling_step=1000 \
logger.wandb.offline=false \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
#debug=short \
