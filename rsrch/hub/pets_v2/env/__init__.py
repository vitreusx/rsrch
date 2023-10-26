from .config import Config
from . import atari, dmc, gym


def make_factory(cfg: Config):
    if cfg.type == "atari":
        return atari.Factory(cfg.atari)
    elif cfg.type == "dmc":
        return dmc.Factory(cfg.dmc)
    elif cfg.type == "gym":
        return gym.Factory(cfg.gym)
    else:
        raise ValueError(f"Invalid env type: {cfg.type}")
