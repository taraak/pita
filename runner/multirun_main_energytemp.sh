#!/bin/bash
#SBATCH -J lj13_energy_temp_a100           # Job name
#SBATCH -o watch_folder/%x_%j.out      # output file (%j expands to jobID)
#SBATCH -N 1                           # Total number of nodes requested
#SBATCH --get-user-env                 # retrieve the users login environment
#SBATCH --mem=24G                      # server memory requested (per node)
#SBATCH -t 24:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=main               # Request partition
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100l:2             # Type/number of GPUs needed
#SBATCH -c 4
#SBATCH --open-mode=append             # Do not overwrite logs
#SBATCH --requeue                      # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@120

eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

export seed=62;

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u src/train.py -m model=energytemp experiment=lj13_energytemp trainer=ddp model.resampling_interval=1 tags=["test","LJ13"]
