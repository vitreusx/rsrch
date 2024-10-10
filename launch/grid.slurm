#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd ~/self/rsrch
eval "$(~/miniforge3/bin/conda shell.bash hook)"
source $(~/.local/bin/poetry env info --path)/bin/activate

LINENO=$(($SLURM_ARRAY_TASK_ID + 1))
eval "$(sed "${LINENO}q;d" grid.txt)"
