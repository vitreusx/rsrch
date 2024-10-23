from typing import Sequence

import gymnasium


class StackSeq(Sequence):
    def __init__(
        self,
        seq: Sequence,
        stack_num: int,
        span: range | None = None,
    ):
        super().__init__()
        self.seq = seq
        self.stack_num = stack_num
        if span is None:
            span = range(0, len(self.seq))
        self.span = span

    def __len__(self):
        return len(self.span)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return StackSeq(self.seq, self.stack_num, self.span[idx])
        else:
            idx = self.span[idx]

            if idx < self.stack_num - 1:
                xs = [
                    *(self.seq[0] for _ in range(self.stack_num - 1 - idx)),
                    *self.seq[: idx + 1],
                ]
            else:
                xs = self.seq[idx - self.stack_num + 1 : idx + 1]

            obs = tuple(x["obs"] for x in xs)
            return {**self.seq[idx], "obs": obs}


class MapSeq(Sequence):
    def __init__(self, seq: Sequence, f):
        super().__init__()
        self.seq = seq
        self.f = f

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MapSeq(self.seq[idx], self.f)
        else:
            return self.f(self.seq[idx])


class GymnasiumFrameSkip(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, frame_skip: int = 1):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward = None
        for _ in range(self.frame_skip):
            next_obs, reward, term, trunc, info = super().step(action)
            if total_reward is None:
                total_reward = reward
            else:
                total_reward += reward
            if term or trunc:
                break
        return next_obs, total_reward, term, trunc, info


class GymnasiumRecordStats(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._total_steps = 0
        self._ep_returns = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._total_steps += 1
        self._ep_length += 1
        info["total_steps"] = self._total_steps
        return obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self._total_steps += 1
        info["total_steps"] = self._total_steps
        self._ep_length += 1
        self._ep_returns += reward
        if term or trunc:
            info["ep_returns"] = self._ep_returns
            self._ep_returns = 0.0
            info["ep_length"] = self._ep_length
            self._ep_length = 0
        return next_obs, reward, term, trunc, info
