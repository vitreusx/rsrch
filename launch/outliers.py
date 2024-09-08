import argparse
import shlex
from itertools import product

from .common import *


def outliers(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.train"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    seeds = [*range(4)]
    envs = ["Assault", "BattleZone"]

    for seed, env in product(seeds, envs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}{suffix}",
            **common_opts,
        }
        args = [*common_args, "-o", dumps(options)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="outliers")
    args = p.parse_args()

    all_tests = outliers(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
