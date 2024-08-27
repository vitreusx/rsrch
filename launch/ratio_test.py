import argparse
import shlex
from itertools import product

from .common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-s",
        "--slice",
        type=str,
        help="Slice specifier, in the form of <slice 1-index>/<num of slices> (e.g. 1/4, 2/4, .., 4/4)",
    )
    args = p.parse_args()

    seeds = [*range(3)]
    envs = ATARI_100k_5
    freqs = [64, 24, 4]

    grid = [*product(seeds, envs, freqs)]
    if args.slice is not None:
        index, total = args.slice.split("/")
        index, total = int(index, 10), int(total, 10)
        start = int((index - 1) * len(grid) / total)
        end = int(index * len(grid) / total)
        grid = grid[start:end]

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

    for seed, env, freq in grid:
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
