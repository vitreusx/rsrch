import argparse
import shlex
from itertools import product
from pathlib import Path

from .utils import *


def pretrain(test_name, suffix=""):
    all_tests = []

    preset_file = Path(__file__).parent / "presets.yml"
    common_args = [
        "-P",
        PRESET_PATH,
        "-p",
        "thesis.pretrain",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO[:1]
    rl_freq_mults = [1.0, 0.0]
    ratios = [8, 4, 2]
    seeds = [*range(5)]

    for env, wm_ratio, rl_ratio, rl_freq_mult, seed in product(
        envs, ratios, ratios, rl_freq_mults, seeds
    ):
        rl_freq = rl_ratio / wm_ratio * rl_freq_mult
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "_wm_ratio": wm_ratio,
            "_rl_ratio": rl_ratio,
            "_rl_freq": rl_freq,
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-wm_ratio={wm_ratio}-rl_ratio={rl_ratio}-rl_freq={rl_freq}-seed={seed}"
            + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pretrain")
    args = p.parse_args()

    all_tests = pretrain(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
