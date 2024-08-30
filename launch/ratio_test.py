import argparse
import shlex
from itertools import product

from .common import *


def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()

    seeds = [*range(3)]
    envs = ATARI_100k_5
    freqs = [64, 24, 4]

    common_args = [
        "python",
        "-m",
        "rsrch.hub.rl.dreamer",
        "-p",
        "atari.base",
        "atari.train",
    ]

    common_opts = {
        "run.no_ansi": True,
        "run.create_commit": False,
    }

    for seed, env, freq in product(seeds, envs, freqs):
        options = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.prefix": f"{env}-seed={seed}-freq={freq}",
            **common_opts,
        }

        args = [*common_args, "-o", dumps(options)]

        print(shlex.join(args))


if __name__ == "__main__":
    main()