#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Activate conda and poetry
eval "$(~/miniforge3/bin/conda shell.bash hook)"
source $(~/.local/bin/poetry env info --path)/bin/activate

# Execute command
eval "$(sed "${SLURM_ARRAY_TASK_ID}q;d" commands.txt)"
