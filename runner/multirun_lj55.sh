#!/bin/bash
#SBATCH -J lj55_dem_h100                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=384G                     # server memory requested (per node)
#SBATCH -t 3:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable        # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100l:4                  # Type/number of GPUs needed
#SBATCH -c 6
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@120


eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

export seed=62;

srun python -u src/train.py -m experiment=lj55 trainer=ddp model.resampling_interval=-1 data.n_train_batches_per_epoch=200 model.num_samples_to_sample_from_buffer=64 model.num_samples_to_generate_per_epoch=64 tags=["oldDEM","LJ55"]

