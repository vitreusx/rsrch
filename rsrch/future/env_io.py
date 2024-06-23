from collections import deque

from rsrch.rl import gym


class EnvIO:
    def observe(self):
        ...

    def step(self, action):
        ...


class _GymEnvIO:
    def __init__(self, env: gym.Env):
        self.env = env
        self._needs_reset = True
        self._actions = deque()

    def __next__(self):
        if self._needs_reset:
            obs, info = self.env.reset()
            self._needs_reset = False
            return {"obs": obs, "info": info}

        action = self._actions.popleft()
        next_obs, reward, term, trunc, info = self.env.step(action)
        self._needs_reset = term or trunc
        return {
            "obs": next_obs,
            "reward": reward,
            "term": term,
            "trunc": trunc,
            "info": info,
        }

    def put(self, action):
        self._actions.append(action)


def GymEnvIO(env: gym.Env):
    _io = _GymEnvIO(env)
    return _io, _io
