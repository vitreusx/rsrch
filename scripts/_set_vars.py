"""This small script fetches env vars from environment.yml and yields KEY=VALUE pairs."""

import argparse
from pathlib import Path
from ruamel.yaml import safe_load
import shlex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--env-file", type=Path, default="environment.yml")

    args = p.parse_args()

    assert args.env_file.exists()
    with open(args.env_file, "r") as env_file:
        env_yml = safe_load(env_file)

    cmd = ["conda", "env", "config", "vars", "set"]
    for key, value in env_yml["variables"].items():
        cmd += [f'{key}="{value}"']

    cmd = shlex.join(cmd)
    print(cmd)


if __name__ == "__main__":
    main()
