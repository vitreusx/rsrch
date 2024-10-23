#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

eval "$(~/miniforge3/bin/conda shell.bash hook)"
source $(~/.local/bin/poetry env info --path)/bin/activate

eval "$(sed "${SLURM_ARRAY_TASK_ID}q;d" commands.txt)"
