#!/bin/bash
#SBATCH -J lj13_samples
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --array=1-100
#SBATCH --mem=2G
#SBATCH -t 48:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long-cpu
#SBATCH -c 1
#SBATCH --open-mode=append            # Do not overwrite logs
echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID"

eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv


TEMPERATURE=0.9
NOISE_STD=0.05
python sample_lj.py --temperature ${TEMPERATURE} --num_burnin_steps 5000 --save_file ${SCRATCH}/lj13_samples/samples_${NOISE_STD}_noise_${TEMPERATURE}_${SLURM_ARRAY_TASK_ID}.npy
