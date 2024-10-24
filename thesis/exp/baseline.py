from itertools import product
from pathlib import Path
from .utils import *

import argparse
import shlex


def baseline(test_name, suffix=""):
    all_tests = []

    preset_file = Path(__file__).parent / "presets.yml"
    common_args = [
        "-P",
        str(preset_file.relative_to(Path.cwd())),
        "-p",
        "thesis.baseline",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO
    ratios = [64, 32, 16, 8, 4, 2]
    seeds = [*range(5)]

    for env, ratio, seed in product(envs, ratios, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_ratio": ratio,
            "run.dir": f"runs/{test_name}/{env}-ratio={ratio}-seed={seed}" + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    args = p.parse_args()

    all_tests = baseline(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
