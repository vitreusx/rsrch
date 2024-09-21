import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def true_sanity_check(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.train"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    seeds = [*range(5)]
    envs = [*{*ATARI_100k_5, *A100k_STABLE_5}]

    for seed, env in product(seeds, envs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}{suffix}",
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(options)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="true-sanity-check")
    args = p.parse_args()

    all_tests = true_sanity_check(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()