#!/bin/bash
#SBATCH --account=aip-bengioy
#SBATCH --mem-per-cpu=24G
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00

module purge
module load python/3.11 cuda/12.2
module load openmm/8.2.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r cc_requirements.txt
cd runner/
pip install -e .

#HYDRA_FULL_ERROR=1 srun python -u src/train.py \
HYDRA_FULL_ERROR=1 python -u src/train.py \
+trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp \
trainer=ddp model.resampling_interval=1 \
tags=["test","ALDP"] \
model.noise_schedule.sigma_min=0.05 \
model.dem.num_training_epochs=0 \
logger.wandb.offline=true \
