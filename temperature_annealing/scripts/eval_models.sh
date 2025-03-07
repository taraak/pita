#!/bin/bash

path="/network/scratch/t/tara.akhoundsadegh/dem/logs/train/runs/2024-01-28_14-52-34/active-inference/psr21x27/checkpoints/epoch\=999-step\=100000.ckpt"

python src/eval.py experiment=dw4 ckpt_path=$path hydra.job.chdir=True model.num_samples_to_save=100000
