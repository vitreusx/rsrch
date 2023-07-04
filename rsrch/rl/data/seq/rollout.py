import itertools

from rsrch.rl import agents, gym
from rsrch.utils import data

from .data import ListSeq


class SeqRollout(data.IterableDataset[ListSeq]):
    def __init__(self, env: gym.Env, agent: agents.Agent, num_episodes=None):
        super().__init__()
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes

    def __iter__(self):
        for ep_idx in itertools.count():
            if self.num_episodes is not None:
                if ep_idx >= self.num_episodes:
                    return

            obs, _ = self.env.reset()
            self.agent.reset()

            # The zero in reward is there to make indexing standard
            ep = ListSeq(obs=[obs], act=[], reward=[], term=[False])

            while True:
                act = self.agent.act(obs)
                next_obs, reward, term, trunc, _ = self.env.step(act)

                ep.obs.append(next_obs)
                ep.act.append(act)
                ep.reward.append(reward)
                ep.term.append(term)

                obs = next_obs
                if term or trunc:
                    break

            yield ep
