#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue

# Activate conda and poetry
eval "$(~/miniforge3/bin/conda shell.bash hook)"
source $(~/.local/bin/poetry env info --path)/bin/activate

# Execute command. We attempt 3 retries (sometimes jobs fail due to SIGSEGV or random CUDA errors.)
COMMAND="$(sed "${SLURM_ARRAY_TASK_ID}q;d" commands.txt)"
for iter in {1..3}; do
    eval "${COMMAND}"
    if [ $? -eq 0 ]; then
        break
    fi
done
