#!/bin/bash
#SBATCH -J lj13_tempdem_h100          # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=384G                    # server memory requested (per node)
#SBATCH -t 3:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=short-unkillable  # Request partition
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100l:4            # Type/number of GPUs needed
#SBATCH -c 6
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --signal=SIGUSR1@120


eval "$(micromamba shell hook --shell bash)"
micromamba activate ~/scratch/demenv

export seed=62;

# To enable preemption re-loading, set `hydra.run.dir` or
srun python -u src/train.py -m model=tempdem experiment=lj13_temp4 trainer=ddp model.resampling_interval=1 energy.temperature=2.0 model.annealed_energy.temperature=1.0 tags=["tempDEM","LJ13"]

