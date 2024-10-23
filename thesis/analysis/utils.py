from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
from ruamel.yaml import YAML
from scipy.signal import lfilter, lfiltic
from tbparse import SummaryReader

pio.kaleido.scope.mathjax = None

yaml = YAML(typ="safe", pure=True)


class TBScalars:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def read(self, test_dir: str | Path):
        test_dir = Path(test_dir)

        dst_path = self.cache_dir / f"{test_dir.name}.h5"
        if not dst_path.exists():
            board = SummaryReader(test_dir / "board")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            board.scalars.to_hdf(dst_path, key="scalars")

        return pd.read_hdf(dst_path, key="scalars")


def exp_mov_avg(arr: np.ndarray, alpha: float):
    b, a = [1.0 - alpha], [1.0, -alpha]
    zi = lfiltic(b, a, arr[:1], [0])
    return lfilter(b, a, arr, zi=zi)[0]


ATARI_100k = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Bank Heist",
    "Battle Zone",
    "Boxing",
    "Breakout",
    "Chopper Command",
    "Crazy Climber",
    "Demon Attack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "James bond",
    "Kangaroo",
    "Krull",
    "Kung Fu Master",
    "Ms Pacman",
    "Pong",
    "Private Eye",
    "Qbert",
    "Road Runner",
    "Seaquest",
    "Up N Down",
]
