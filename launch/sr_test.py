import argparse
import io
import shlex
from itertools import product

from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


def dumps(data):
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


from .common import ATARI_100k_5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    seeds = [*range(5)]
    envs = ATARI_100k_5
    freqs = [64, 24, 4]

    common_args = [
        "python",
        "-m",
        "rsrch.hub.rl.dreamer",
        "-p",
        "atari.base",
        "atari.train_sr",
    ]

    common_ov = {
        "run.no_ansi": True,
    }

    for seed, env, freq in product(seeds, envs, freqs):
        overrides = {
            "env": {"type": "atari", "atari.env_id": env},
            "repro.seed": seed,
            "_freq": freq,
            "run.prefix": f"{env}-seed={seed}-freq={freq}",
            **common_ov,
        }

        args = [*common_args, "-o", dumps(overrides)]

        print(shlex.join(args))


if __name__ == "__main__":
    main()
