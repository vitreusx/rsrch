import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def low_ratio_test(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.train_val"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    seeds = [0]
    envs = ATARI_100k_5
    freqs = [2, 1]

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
    p.add_argument("--name", default="low-ratio-test")
    args = p.parse_args()

    all_tests = [
        *sanity_check(args.name, "-sanity"),
        *low_ratio_test(args.name),
    ]

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
