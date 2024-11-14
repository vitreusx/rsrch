import argparse
import shlex
from itertools import product
from pathlib import Path

from .utils import *


def sanity_check(test_name, suffix=""):
    all_tests = []

    preset_file = Path(__file__).parent / "presets.yml"
    common_args = [
        "-P",
        PRESET_PATH,
        "-p",
        "thesis.sanity_check",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]

    for env, seed in product(envs, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sanity_check")
    args = p.parse_args()

    all_tests = sanity_check(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
