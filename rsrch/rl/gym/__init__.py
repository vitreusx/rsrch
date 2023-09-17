from . import agents, spaces, vector, wrappers
from .agents import Agent
from .base import *
from .spaces import TensorSpace
from .vector import Agent as VecAgent
from .vector import VectorEnv


def is_vector_env(env):
    return getattr(env, "is_vector_env", False)
