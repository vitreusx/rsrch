import argparse
import shlex
from itertools import product
from pathlib import Path

from ..utils import *


def data_aug_v2(test_name, suffix=""):
    all_tests = []

    preset_file = Path(__file__).parent / "presets.yml"
    common_args = [
        "-P",
        PRESET_PATH,
        "-p",
        "thesis.data_aug.v2",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO
    ratios = [2, 4]
    seeds = [*range(5)]

    for env, ratio, seed in product(envs, ratios, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "_ratio": ratio,
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-ratio={ratio}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="data_aug/v2")
    args = p.parse_args()

    all_tests = data_aug_v2(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
