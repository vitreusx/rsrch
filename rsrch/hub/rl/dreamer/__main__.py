import argparse
from pathlib import Path

from rsrch.exp import tensorboard

from . import config


def main():
    presets = []

    cfg = config.cli(
        cls=config.Config,
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
        args=argparse.Namespace(presets=presets),
    )

    ...
