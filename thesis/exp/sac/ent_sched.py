import argparse
import shlex
from itertools import product
from pathlib import Path

import numpy as np

from ..utils import *


def sac_ent_sched(test_name: str, suffix=""):
    all_tests = []

    common_args = [
        "-P",
        "thesis/exp/presets.yml",
        "-p",
        "thesis.sac.ent_sched",
        "grid_launch",
    ]

    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]

    for env, seed in product(envs, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_ratio": 4,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sac/ent_sched")
    args = p.parse_args()

    all_tests = sac_ent_sched(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
