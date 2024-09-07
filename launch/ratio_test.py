import argparse
import shlex
from itertools import product

from .common import *


def ratio_test(test_name, suffix=""):
    seeds = [*range(3)]
    envs = ATARI_100k_5
    freqs = [64, 24, 4]

    common_args = ["-p", "atari.base", "atari.train"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    all_tests = []
    for seed, env, freq in product(seeds, envs, freqs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-freq={freq}",
            **common_opts,
        }
        all_tests.append([*common_args, "-o", dumps(options)])

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ratio-test")
    args = p.parse_args()

    all_tests = ratio_test(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
