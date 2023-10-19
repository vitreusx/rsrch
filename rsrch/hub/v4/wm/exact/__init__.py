from . import cartpole
from .core import Agent, Trainer


def from_id(env_id: str):
    if env_id == "CartPole-v1":
        return cartpole
