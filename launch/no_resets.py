import argparse
import io
import shlex
from itertools import product

from ruamel.yaml import YAML

from .common import ATARI_100k_3

yaml = YAML(typ="safe", pure=True)
yaml.default_flow_style = True
yaml.width = int(2**10)


def dumps(data):
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue().rstrip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seeds = [*range(3)]
    envs = ATARI_100k_3
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
