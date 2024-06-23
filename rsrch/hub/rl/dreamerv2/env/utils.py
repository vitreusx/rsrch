from . import atari, base, gym
from .config import Config


def make_factory(cfg: Config, device=None, seed=None) -> base.Factory:
    if cfg.type == "atari":
        return atari.Factory(cfg.atari, device, seed)
    elif cfg.type == "gym":
        return gym.Factory(cfg.gym, device, seed)
    else:
        raise ValueError(f"Invalid env type: {cfg.type}")
