from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
            board = SummaryReader(test_dir / "board", extra_columns={"wall_time"})
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            board.scalars.to_hdf(dst_path, key="scalars")

        return pd.read_hdf(dst_path, key="scalars")


def exp_mov_avg(arr: np.ndarray, alpha: float):
    b, a = [1.0 - alpha], [1.0, -alpha]
    zi = lfiltic(b, a, arr[:1], np.zeros((1,), dtype=arr.dtype))
    return lfilter(b, a, arr, zi=zi)[0]


ATARI_100k = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "ChopperCommand",
    "CrazyClimber",
    "DemonAttack",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MsPacman",
    "Pong",
    "PrivateEye",
    "Qbert",
    "RoadRunner",
    "Seaquest",
    "UpNDown",
]


def to_rgba(desc: str, alpha: float = 0.2):
    r, g, b = eval(desc.removeprefix("rgb"))
    return f"rgba{(r, g, b, alpha)}"


def err_line(x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str, **kwargs):
    y_lower, y_upper = y - std, y + std
    return [
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color),
            **kwargs,
        ),
        go.Scatter(
            x=[*x, *x[::-1]],
            y=[*y_upper, *y_lower[::-1]],
            fill="tozerox",
            fillcolor=to_rgba(color),
            line=dict(color="rgba(255, 255, 255, 0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
    ]


def make_color_iter(palette="Plotly"):
    while True:
        for color in getattr(px.colors.qualitative, palette):
            if color.startswith("#"):
                color = color.removeprefix("#")
                r, g, b = color[0:2], color[2:4], color[4:6]
                r, g, b = (int(v, 16) for v in (r, g, b))
                yield f"rgb{(r, g, b)}"
            elif color.startswith("rgb("):
                yield color


def load_config(test_dir: Path):
    with open(test_dir / "config.yml", "r") as f:
        return yaml.load(f)


def config_value(cfg: dict, key: str):
    parts = key.split(".")
    for part in parts:
        cfg = cfg[part]
    return cfg
