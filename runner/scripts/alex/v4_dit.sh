#!/bin/bash

#SBATCH -J aldp_dit_v2            # Job name
#SBATCH -o watch_folder/%x_%A_%a.out            # Output file (%x=job name, %j=job ID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH --get-user-env                       # Retrieve the user's login environment
#SBATCH --mem=24G                            # Memory per node
#SBATCH -t 3:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --gres=gpu:a100l:4
#SBATCH --array=[0-10%1]
#SBATCH -c 4                                 # Number of CPU cores
#SBATCH --open-mode=append                   # Append to logs instead of overwriting
#SBATCH --requeue                            # Requeue upon pre-emption

RUN_NAME="dit_good_v6_l40"
HYDRA_FULL_ERROR=1 python src/train.py +trainer.num_sanity_val_steps=0 \
model=energytemp \
experiment=alp_energytemp_good \
trainer=ddp \
tags=["test","ALDP","v4"] \
trainer.check_val_every_n_epoch=100 \
data.n_train_batches_per_epoch=125 \
trainer.max_epochs=1000 \
model.debias_inference=True \
model.training_batch_size=2048 \
model.inference_batch_size=1024 \
model.num_samples_to_save=4096 \
model.num_negative_time_steps=0 \
++model.compile=True \
++model.train_on_all_temps=False
hydra.run.dir='${paths.log_dir}/${task_name}/runs/'${RUN_NAME} \
ckpt_path='${paths.log_dir}/${task_name}/runs/'${RUN_NAME}/checkpoints/last.ckpt \
#ckpt_path=/network/scratch/t/tara.akhoundsadegh/energy_temp/logs/train/runs/2025-05-05_04-40-47/checkpoints/epoch_299.ckpt \

#
#model.net.hidden_size=768 \
#model.net.cond_dim=64 \
#model.net.n_blocks=6 \
#model.net.n_heads=6 \
#logger.wandb.id=${RUN_NAME}
#model.num_samples_to_save=5000 \
#debug=short \
#model/net=egnn_dynamics_ad2_cat \
#++model.debug_fm=True \
#model.inference_batch_size=384 \
#model.net.hidden_nf=64 \
#model.net.n_layers=5 \
#model/net=dit \
#model.end_resampling_step=900 \
#model.net.hidden_nf=64 \
#model/net=egnn_dynamics_ad2_cat \
#++model.compile=True
#trainer.gradient_clip_val=100 \
#model.do_langevin=true \
#debug=short \
#+model.only_train_score=True \
#+energy.debug_train_on_test=True \
