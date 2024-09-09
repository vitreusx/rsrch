import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def primed_wm_test(test_name, suffix="", fast=False):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.primed_wm"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    if fast:
        seeds = [0]
        envs = A100k_MONO[:1]
        freq = [4, 8]
    else:
        seeds = [0]
        envs = A100k_MONO[:3]
        freq = [2, 4, 8]

    for seed, env, wm_freq, ac_freq in product(seeds, envs, freq, freq):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-wm_freq={wm_freq}-ac_freq={ac_freq}{suffix}",
            "_wm_freq": wm_freq,
            "_ac_freq": ac_freq,
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(options)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="primed-wm-test")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    all_tests = primed_wm_test(args.name, fast=args.fast)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
