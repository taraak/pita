#!/bin/bash
#SBATCH -J lj13_v23
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --array=1-30                  # array job
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=2G
#SBATCH -t 2-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=long-cpu
#SBATCH -c 1
#SBATCH --open-mode=append            # Do not overwrite logs
echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID"

# Define arrays for temperatures and kernels
declare -a temps=(1.2 1.5 1.8)
#declare -a temps=(0.2 0.3 0.4 0.5 0.6)
declare -a kernels=("nuts")
#declare -a kernels=("randomwalk")

# Define the number of seeds
num_seeds=10

# Get the current index for array job
idx=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate the position in the combined list of temperatures and kernels
temp_idx=$((idx / (num_seeds * ${#kernels[@]})))
kernel_idx=$(((idx % (num_seeds * ${#kernels[@]})) / num_seeds))
seed_idx=$((idx % num_seeds))

temperature=${temps[$temp_idx]}
kernel=${kernels[$kernel_idx]}
seed=$((seed_idx + 1))

echo "Starting job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID with temperature ${temperature} kernel $kernel and seed $seed"

# Execute your program with the current temperature and kernel
python sample_lj13.py --save_file ${SCRATCH}/lj13_samples/samples_v23_${temperature}_${SLURM_ARRAY_TASK_ID}.npy \
                      --warmup_steps 20000 \
                      --num_samples 20000 \
                      --temperature ${temperature} \
                      --kernel $kernel

# v20 is RW with 20k warmup and 20k steps
# v21 is RW with 200k warmup and 20k steps
# v22 is NUTS with 20k warmup and 20k steps
# v23 is NUTS with 20k warmup and 20k steps

#TEMPERATURE=1.0
#WARMUP_STEPS=200000
# v2 has 2k warmup
# v4 has 10k warmup
# v5 has 20k warmup
# v6 has 220k total steps
# v7 has 10k total steps
# v8 has 2k warmup
# v9 is Gaussian and 200k warmup with 20k steps
# v11 is updated code 2000 warmup and 20k steps
# v12 is updated code with random walk kernel
# v13 is updated code with random walk kernel and 20k warmup
# v14 is updated code with random walk kernel and 200k warmup
# v15 is updated code with 20k warmup
# v16 is updated code with 200k warmup
#python sample_lj13.py --save_file ${SCRATCH}/lj13_samples/samples_v16_${TEMPERATURE}_${SLURM_ARRAY_TASK_ID}.npy --warmup_steps ${WARMUP_STEPS} --num_samples 20000 --temperature ${TEMPERATURE} # --kernel randomwalk
#python sample_lj13.py --save_file ${SCRATCH}/lj13_samples/samples_v9_${TEMPERATURE}_${SLURM_ARRAY_TASK_ID}.npy --warmup_steps ${WARMUP_STEPS} --num_samples 20000 --temperature ${TEMPERATURE} --kernel randomwalk
