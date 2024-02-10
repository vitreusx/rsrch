from .base import *  # isort: skip
from . import agents, envs, vector, wrappers
from .agents import Agent
from .vector import Agent as VecAgent
from .vector import VectorEnv


def is_vector_env(env):
    return getattr(env, "is_vector_env", False)
