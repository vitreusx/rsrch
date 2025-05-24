#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue

# Activate conda and poetry
CONDA="${CONDA:-${HOME}/miniforge3/bin/conda}"
eval "$("${CONDA}" shell.bash hook)"

POETRY="${POETRY:-${HOME}/.local/bin/poetry}"
source $("${POETRY}" env info --path)/bin/activate

# Extract the command from the commands.txt file
COMMAND="$(sed "${SLURM_ARRAY_TASK_ID}q;d" commands.txt)"

# Extract the run directory from the config
RUN_DIR="$(eval "${COMMAND} --dump-config" | jq -r '.run.dir')"

# If the run directory exists, we stop the job, since run directories cannot be overwritten.
if [ -d "${RUN_DIR}" ]; then
    exit 0
fi

# Try to execute command. We attempt 3 retries (sometimes jobs fail due to SIGSEGV or random CUDA errors.)
for retry in {1..3}; do
    eval "${COMMAND}"
    if [ $? -eq 0 ]; then
        break
    else
        rm -rf "${RUN_DIR}"
    fi
done
