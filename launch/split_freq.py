import argparse
import shlex
from itertools import product

from .common import *
from .sanity_check import sanity_check


def split_freq(test_name: str, suffix: str = ""):
    common_args = ["-p", "exp.split_freq", "grid_launch"]

    envs = ["Assault"]
    seeds = [*range(3)]
    wm_freqs = [1, 2, 4]
    rl_freqs = [1, 2, 4]
    grid = product(envs, seeds, wm_freqs, rl_freqs)

    all_tests = []
    for env, seed, wm_freq, rl_freq in grid:
        opts = {
            "env.atari.env_id": env,
            "repro.seed": seed,
            "_wm_freq": wm_freq,
            "_rl_freq": rl_freq,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-wm_freq={wm_freq}-rl_freq={rl_freq}{suffix}",
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-name", default="split-freq")
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    all_tests = split_freq(args.test_name)
    if args.sanity:
        all_tests.extend(sanity_check(args.test_name, "-sanity"))

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
