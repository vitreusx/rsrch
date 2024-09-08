import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def primed_wm_test(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "atari.base", "atari.primed_wm"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    seeds = [0]
    envs = A100k_MONO[:3]
    freq = [2, 4, 8]

    for seed, env, wm_freq, ac_freq in product(seeds, envs, freq, freq):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}{suffix}",
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
    args = p.parse_args()

    all_tests = primed_wm_test(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
