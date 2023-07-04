import itertools

from rsrch.rl import agents, gym
from rsrch.utils import data

from .data import Step


class StepRollout(data.IterableDataset[Step]):
    def __init__(
        self, env: gym.Env, agent: agents.Agent, num_steps=None, num_episodes=None
    ):
        super().__init__()
        self.env = env
        self.agent = agent
        self.num_steps = num_steps
        self.num_episodes = num_episodes

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        total_steps = 0

        for ep_idx in itertools.count():
            if self.num_episodes is not None:
                if ep_idx >= self.num_episodes:
                    return

            obs = None

            for _ in itertools.count():
                if self.num_steps is not None:
                    if total_steps >= self.num_steps:
                        return

                if obs is None:
                    obs, _ = self.env.reset()
                    self.agent.reset()

                act = self.agent.act(obs)
                next_obs, reward, term, trunc, _ = self.env.step(act)
                yield Step(obs, act, next_obs, reward, term)
                obs = next_obs
                total_steps += 1

                if term or trunc:
                    break
