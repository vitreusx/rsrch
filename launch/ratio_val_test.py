import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def ratio_val_test(suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.train_val"]
    common_opts = {
        "run.no_ansi": True,
        "run.create_commit": False,
    }

    seeds = [0]
    envs = ATARI_100k_3
    freqs = [64, 32, 16, 8, 4]

    for seed, env, freq in product(seeds, envs, freqs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.prefix": f"{env}-seed={seed}-freq={freq}{suffix}",
            **common_opts,
        }
        args = [*common_args, "-o", dumps(options)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()

    all_tests = [*sanity_check("-sanity"), *ratio_val_test()]
    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
