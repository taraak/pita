#!/bin/bash
#SBATCH -J lj13
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --array=1-100
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=2G
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long-cpu
#SBATCH -c 1
#SBATCH --open-mode=append            # Do not overwrite logs
echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID"

TEMPERATURE=4
python sample_lj13.py --save_file ${SCRATCH}/lj13_samples/samples_v2_${TEMPERATURE}_${SLURM_ARRAY_TASK_ID}.npy
