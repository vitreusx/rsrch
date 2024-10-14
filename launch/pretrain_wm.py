import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def pretrain_wm(test_name: str, suffix: str = ""):
    common_args = ["-p", "exp.pretrain_wm", "grid_launch"]

    envs = A100k_MONO[:3]
    seeds = [*range(3)]
    freqs = [2, 4, 8, 16]
    grid = product(envs, seeds, freqs)

    all_tests = []
    for env, seed, freq in grid:
        opts = {
            "env.atari.env_id": env,
            "repro.seed": seed,
            "_freq": freqs,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-freq={freq}{suffix}",
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-name", default="pretrain-wm")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    all_tests = pretrain_wm(args.test_name)
    if args.sanity:
        all_tests.extend(sanity_check(args.test_name, "-sanity"))

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
