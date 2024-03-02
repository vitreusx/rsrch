from collections import OrderedDict
from functools import partial

import numpy as np
from dm_control import suite
from dm_env.specs import Array, BoundedArray, DiscreteArray

from .base import Env, register, spaces


def spec_to_space(spec):
    if isinstance(spec, DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, BoundedArray):
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, Array):
        return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif isinstance(spec, OrderedDict):
        subspaces = [(key, spec_to_space(value)) for key, value in spec.items()]
        return spaces.Dict(spaces=subspaces)
    else:
        raise ValueError()


class DMCEnv(Env):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        render_mode=None,
        render_size=None,
    ):
        super().__init__()
        self._env = suite.load(domain_name, task_name)
        self.render_mode = render_mode
        self._render_size = render_size
        self.observation_space = spec_to_space(self._env.observation_spec())
        self.action_space = spec_to_space(self._env.action_spec())
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": int(1.0 / self._env.control_timestep()),
        }

    def reset(self, *, seed=None, options=None):
        ts = self._env.reset()
        return ts.observation, {}

    def step(self, action):
        ts = self._env.step(action)
        return ts.observation, ts.reward, ts.last(), False, {}

    def render(self):
        if self._render_size is not None:
            width, height = self._render_size
            return self._env.physics.render(height=height, width=width)
        else:
            return self._env.physics.render()

    def close(self):
        self._env.close()


def register_envs():
    for domain, task in suite.ALL_TASKS:
        domain_ = domain.replace("_", " ").title().replace(" ", "")
        task_ = task.replace("_", " ").title().replace(" ", "")
        env_id = f"DMC/{domain_}-{task_}"
        register(env_id, partial(DMCEnv, domain, task))
