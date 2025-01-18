#!/bin/bash

eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

#path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/multiruns/2025-01-15_18-25-08/0/DEM-2/y7aan072/checkpoints/epoch\=329-step\=8250.ckpt"
path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/multiruns/2025-01-17_15-09-20/0/DEM-2/yk8tduna/checkpoints/epoch\=329-step\=8250.ckpt"

TEMPERATURE=2.0

# model.num_samples_to_generate_per_epoch_energy=1024
# To enable preemption re-loading, set `hydra.run.dir` or
python src/eval.py -m launcher=mila_a100 \
ckpt_path=$path \
model=tempdem experiment=lj13_temp4 \
model.resampling_interval=1 energy.temperature=${TEMPERATURE} model.annealed_energy.temperature=0.7,0.8,1.0,2.0 \
tags=["eval_sweep","LJ13","tempDEM"]
