from . import atari, dmc, gym
from .config import Config


def make_factory(cfg: Config, device=None):
    if cfg.type == "atari":
        return atari.Factory(cfg.atari, device)
    elif cfg.type == "dmc":
        return dmc.Factory(cfg.dmc, device)
    elif cfg.type == "gym":
        return gym.Factory(cfg.gym, device)
    else:
        raise ValueError(f"Invalid env type: {cfg.type}")