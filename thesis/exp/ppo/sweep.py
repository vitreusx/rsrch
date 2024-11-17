import argparse
import shlex
from itertools import product
from pathlib import Path

import numpy as np

from ..utils import *


def ppo_sweep(test_name: str, suffix=""):
    all_tests = []

    common_args = [
        "-P",
        "thesis/exp/presets.yml",
        "-p",
        "thesis.ppo.base",
        "grid_launch",
    ]

    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]
    configs = {
        "k8_e4_mb1": {
            "rl_ratio": 8,
            "update_epochs": 4,
            "num_minibatches": 1,
        },
        "k8_e4_mb8": {
            "rl_ratio": 8,
            "update_epochs": 4,
            "num_minibatches": 8,
        },
        "k16_e8_mb1": {
            "rl_ratio": 16,
            "update_epochs": 8,
            "num_minibatches": 1,
        },
        "k8_e8_mb1": {
            "rl_ratio": 8,
            "update_epochs": 8,
            "num_minibatches": 1,
        },
    }

    for env, seed, (name, config) in product(envs, seeds, configs.items()):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_wm_ratio": 4,
            "_rl_ratio": config["rl_ratio"],
            "rl.ppo": {
                "update_epochs": config["update_epochs"],
                "num_minibatches": config["num_minibatches"],
            },
            "run.dir": f"runs/{test_name}/{env}-cfg={name}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ppo/sweep")
    args = p.parse_args()

    all_tests = ppo_sweep(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
