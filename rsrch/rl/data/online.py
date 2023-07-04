import rsrch.utils.data as data
from rsrch.rl.data.trajectory import ListTrajectory
from rsrch.rl.data.step import Step
from rsrch.rl.agent import Agent
import itertools
import gymnasium as gym


class StepRollout(data.IterableDataset[Step]):
    def __init__(self, env: gym.Env, agent: Agent, num_steps=None, num_episdoes=None):
        super().__init__()
        self.env = env
        self.agent = agent
        self.num_steps = num_steps
        self.num_episdoes = num_episdoes

    def __iter__(self):
        total_steps = 0

        for ep_idx in itertools.count():
            if self.num_episdoes is not None:
                if ep_idx >= self.num_episdoes:
                    return

            obs = None

            for step_idx in itertools.count():
                if self.num_steps is not None:
                    if total_steps >= self.num_steps:
                        return

                if obs is None:
                    obs, info = self.env.reset()
                    self.agent.reset()

                act = self.agent.act(obs)
                next_obs, reward, term, trunc, info = self.env.step(act)
                yield Step(obs, act, next_obs, reward, term, trunc)
                obs = next_obs
                total_steps += 1

                if term or trunc:
                    break


class EpisodeRollout(data.IterableDataset[ListTrajectory]):
    def __init__(self, env: gym.Env, agent: Agent, num_episodes=None):
        super().__init__()
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def __iter__(self):
        for ep_idx in itertools.count():
            if self.num_episodes is not None:
                if ep_idx >= self.num_episodes:
                    return

            obs, info = self.env.reset()
            self.agent.reset()

            # The zero in reward is there to make indexing standard
            ep = ListTrajectory(
                obs=[obs], act=[], reward=[0.0], term=False, trunc=False
            )

            while True:
                act = self.agent.act(obs)
                next_obs, reward, term, trunc, info = self.env.step(act)

                ep.obs.append(next_obs)
                ep.act.append(act)
                ep.reward.append(reward)
                ep.term = term
                ep.trunc = trunc

                obs = next_obs
                if term or trunc:
                    break

            yield ep
