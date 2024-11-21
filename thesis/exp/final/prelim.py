import argparse
import shlex
from itertools import product

from ..utils import *


def final_prelim(test_name: str, suffix=""):
    all_tests = []

    common_args = [
        "-P",
        "thesis/exp/presets.yml",
        "-p",
        "thesis.final.prelim",
        "grid_launch",
    ]

    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]
    configs = [(8, 8), (16, 8), (4, 8), (8, 4)]

    for env, seed, (rl_ratio, num_epochs) in product(envs, seeds, configs):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_rl_ratio": rl_ratio,
            "_update_epochs": num_epochs,
            "run.dir": f"runs/{test_name}/{env}-cfg={rl_ratio}x{num_epochs}-seed={seed}"
            + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="final/prelim")
    args = p.parse_args()

    all_tests = final_prelim(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
