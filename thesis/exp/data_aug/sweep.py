import argparse
import shlex
from itertools import product
from pathlib import Path

from ..utils import *


def data_aug_sweep(test_name, suffix=""):
    all_tests = []

    common_args = [
        "-P",
        PRESET_PATH,
        "-p",
        "thesis.data_aug.sweep",
        "grid_launch",
    ]
    common_opts = {}

    envs = A100k_MONO
    seeds = [*range(5)]
    ratio = 8
    drq_configs = {
        "shift4": dict(type="shift", max_shift=4),
        "shift2": dict(type="shift", max_shift=2),
        "cutout": dict(type="cutout", apply_prob=0.5),
        "vflip": dict(type="vflip", apply_prob=0.1),
        "rotate": dict(type="rotate", rotate_deg=5.0),
        "intensity": dict(type="intensity", intensity_scale=5e-2),
    }
    drq_configs = [*drq_configs.items()]

    for env, (drq_type, drq_config), seed in product(envs, drq_configs, seeds):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "_ratio": ratio,
            "_drq_config": drq_config,
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-type={drq_type}-ratio={ratio}-seed={seed}"
            + suffix,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="data_aug/sweep")
    args = p.parse_args()

    all_tests = data_aug_sweep(args.name)

    prefix = ["python", "-m", "dreamerx"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
