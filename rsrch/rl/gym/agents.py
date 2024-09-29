from ._api import *


class RandomAgent(Agent):
    def __init__(self, env: Env):
        super().__init__()
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def policy(self):
        return self.act_space.sample()
