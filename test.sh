#!/usr/bin/bash
set -e

source .venv/bin/activate

for EXP in lm ppo resnet resnet_jax unet; do
    echo "Testing ${EXP}"
    python scripts/${EXP}/train.py --test
done
