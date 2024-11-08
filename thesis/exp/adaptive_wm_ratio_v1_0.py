import argparse
import shlex
from itertools import product
from pathlib import Path

from .utils import *


def adaptive_wm_ratio_v1_0(test_name, suffix=""):
    all_tests = []

    preset_file = Path(__file__).parent / "presets.yml"
    common_args = [
        "-P",
        str(preset_file.relative_to(Path.cwd())),
        "-p",
        "thesis.adaptive_ratio.wm_v1",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]
    rl_ratios = [2]

    for env, rl_ratio, seed in product(envs, rl_ratios, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_rl_ratio": rl_ratio,
            "run.dir": f"runs/{test_name}/{env}-rl_ratio={rl_ratio}-seed={seed}"
            + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="adaptive_wm_ratio_v1_0")
    args = p.parse_args()

    all_tests = adaptive_wm_ratio_v1_0(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
