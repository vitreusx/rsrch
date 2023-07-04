from .q_nets import QNetwork
import torch
from torch import Tensor
from .env_spec import EnvSpec


class GreedyAgent:
    def __init__(self, Q: QNetwork):
        self.Q = Q

    def batch_act(self, obs: Tensor) -> Tensor:
        with torch.no_grad():
            q_vals = self.Q(obs)
        return torch.argmax(q_vals, 1)

    def act(self, obs: Tensor) -> Tensor:
        return self.batch_act(obs.unsqueeze(0)).squeeze(0)


class EpsGreedyAgent:
    def __init__(self, Q: QNetwork, eps: float):
        self.Q = Q
        self.greedy = GreedyAgent(self.Q)
        self.eps = eps

    def batch_act(self, obs: Tensor) -> Tensor:
        rand_p = torch.rand(len(obs), device=obs.device)
        rand_act = torch.randint(self.Q.num_actions, (len(obs),))
        greedy_act = self.greedy.batch_act(obs)
        return torch.where(rand_p < self.eps, rand_act, greedy_act)

    def act(self, obs: Tensor) -> Tensor:
        return self.batch_act(obs.unsqueeze(0)).squeeze(0)


class RandomAgent:
    def __init__(self, env: EnvSpec):
        self.action_space = env.action_space

    def __call__(self, obs):
        return self.action_space.sample()
