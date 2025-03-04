#!/bin/bash

#eval "$(micromamba shell hook --shell bash)"
#micromamba activate ~/scratch/demenv

#path="/home/mila/t/tara.akhoundsadegh/feynman-kac-diffusion/runner/dem/logs/train/multiruns/2025-01-21_12-38-22/0/DEM-2/8jm2mt28/checkpoints/epoch\=239-step\=6000.ckpt"
#path="/home/mila/t/tara.akhoundsadegh/feynman-kac-diffusion/runner/dem/logs/train/multiruns/2025-01-21_03-43-31/0/DEM-2/exutipuq/checkpoints/epoch\=239-step\=6000.ckpt"
# fresh plasma
path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/multiruns/2025-01-23_22-35-33/0/checkpoints/epoch_179.ckpt"

# rosy voice
#path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/multiruns/2025-01-24_06-58-58/0/checkpoints/last.ckpt"


TEMPERATURE=2.0

# To enable preemption re-loading, set `hydra.run.dir` or
python src/eval.py -m launcher=mila_a100 \
ckpt_path=$path \
model=tempdem experiment=lj13_tempdem \
model.noise_schedule.sigma_max=4.0 model.noise_schedule.sigma_min=0.01 model.scale_diffusion=False \
model.num_negative_time_steps=1 model.diffusion_scale=1.0 model.num_eval_samples=50000 \
model.annealed_clipper.max_score_norm=1000 model.annealed_test_batch_size=10000 \
model.resampling_interval=1 model.start_resampling_step=30 model.resampling_strategy="systematic" \
energy.temperature=${TEMPERATURE} model.annealed_energy.temperature=1.0 \
tags=["eval_sweep","LJ13","tempDEM","fresh_plasma","clipped","fixed_test","fixed_resampling_var_steps","start_resampling_gaussian_v2","test"]


#,"start_resampling_gaussian"]
