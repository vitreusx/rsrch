import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def ratio_val_test(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.train_val"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    seeds = [0]
    envs = A100k_MONO[:3]
    freqs = [64, 32, 16, 8, 4, 2]

    for seed, env, freq in product(seeds, envs, freqs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-freq={freq}{suffix}",
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(options)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ratio-val-test")
    args = p.parse_args()

    all_tests = [
        *sanity_check(args.name, "-sanity"),
        *ratio_val_test(args.name),
    ]

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
