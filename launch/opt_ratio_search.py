import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def opt_ratio_search(test_name: str, suffix: str = ""):
    common_args = ["-p", "exp.at_400k", "grid_launch"]

    envs = ["Assault"]
    seeds = [*range(10)]
    freqs = [8, 4, 2]
    grid = product(envs, seeds, freqs)

    all_tests = []
    for env, seed, freq in grid:
        opts = {
            "env.atari.env_id": env,
            "repro.seed": seed,
            "_freq": freq,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-freq={freq}{suffix}",
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-name", default="opt-ratio-search")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    all_tests = opt_ratio_search(args.test_name)
    if args.sanity:
        all_tests.extend(sanity_check(args.test_name, "-sanity"))

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
