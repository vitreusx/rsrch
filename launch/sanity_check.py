import argparse
import shlex

from .common import *


def sanity_check(test_name, suffix=""):
    all_tests = []

    common_args = ["-p", "exp.sanity_check"]
    common_opts = {
        "run.interactive": False,
        "run.create_commit": False,
    }

    env, freq = A100k_MONO[0], 64
    for seed in range(3):
        opts = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.dir": f"runs/{test_name}/{env}-seed={seed}-freq={freq}{suffix}",
            **common_opts,
        }
        args = [*common_args, "-o", format_opts(opts)]
        all_tests.append(args)

    return all_tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="sanity-check")
    args = p.parse_args()

    all_tests = sanity_check(args.name)

    prefix = ["python", "-m", "rsrch.hub.rl.dreamer"]
    for test in all_tests:
        print(shlex.join([*prefix, *test]))


if __name__ == "__main__":
    main()
