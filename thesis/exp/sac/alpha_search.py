import argparse
import shlex
from itertools import product
from pathlib import Path

import numpy as np

from ..utils import *


def sac_alpha_search(test_name: str, gen: np.random.Generator, suffix=""):
    all_tests = []

    common_args = [
        "-P",
        "thesis/exp/presets.yml",
        "-p",
        "thesis.sac.alpha_search",
        "grid_launch",
    ]
    common_opts = {}

    envs_seeds = [*product(A100k_MONO, [*range(5)])]
    alphas = gen.lognormal(np.log(1e-3), np.log(1e-2 / 1e-3), size=len(envs_seeds))
    alphas = alphas.tolist()

    for (env, seed), alpha in zip(envs_seeds, alphas):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_ratio": 4,
            "_alpha": alpha,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sac/alpha_search")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gen = np.random.default_rng(seed=args.seed)

    all_tests = sac_alpha_search(args.name, gen)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
