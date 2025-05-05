#!/bin/bash
#SBATCH --account=aip-siamakx
#SBATCH --mem-per-cpu=24G
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --array=0-3  # Adjust the range to match number of sigma_min values
#SBATCH -o watchfolder/%x_%A_%a.out
# Define different sigma_min values
#SIGMA_MIN_LIST=(0.002 0.005  0.01 0.02)
POST_MCMC_STEPS_LIST=(0 5 10 50)
POST_MCMC_STEPS=${POST_MCMC_STEPS_LIST[$SLURM_ARRAY_TASK_ID]}
#SIGMA_MIN=${SIGMA_MIN_LIST[$SLURM_ARRAY_TASK_ID]}

module purge
module load python/3.11 cuda/12.2
module load openmm/8.2.0
module load arrow
module load httpproxy/1.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r cc_requirements_tara_v2.txt
cd runner/
pip install -e .

sleep $((RANDOM % 20))
HYDRA_FULL_ERROR=1 python -u src/train.py \
+trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp_good \
trainer=ddp model.resampling_interval=1 \
tags=["ALDP","dit","wise_silence","temp_cond"] \
model/net=dit \
model.noise_schedule.sigma_min=0.005 \
model.dem.num_training_epochs=0 \
trainer.check_val_every_n_epoch=100 \
model.dem.num_training_epochs=0 \
model.debias_inference=True \
model.resample_at_end=False \
model.loss_weights.energy_matching=1.0 \
model.loss_weights.energy_score=1.0 \
model.loss_weights.score=1.0 \
model.loss_weights.target_score=0.1 \
model.loss_time_threshold.score=0 \
model.inference_batch_size=512 \
model.num_negative_time_steps=0 \
model.do_langevin=False \
model.post_mcmc_steps=$POST_MCMC_STEPS \
model.loss_time_threshold.score=0 \
model.end_resampling_step=800 \
model.num_temp_annealed_samples_to_generate=10000 \
model.num_epochs_per_temp=[300,300,300,300,300] \
