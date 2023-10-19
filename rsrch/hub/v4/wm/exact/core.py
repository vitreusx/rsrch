import torch
from rsrch.rl import gym
from rsrch.rl.data.core import ChunkBatch


class WorldModel:
    def reset(self, num_envs: int):
        ...

    def step(self, s, a):
        ...

    def term(self, s):
        ...

    def reward(self, next_s):
        ...

    def act_enc(self, a):
        ...

    def act_dec(self, enc_a):
        ...


class Trainer:
    def __init__(self, wm: WorldModel):
        self.wm = wm

    def opt_step(self, batch: ChunkBatch, ctx):
        seq_len, bs = batch.act.shape[:2]
        enc_act = self.wm.act_enc(batch.act.flatten(0, 1))
        enc_act = enc_act.reshape(seq_len, bs, *enc_act.shape[1:])

        states = [batch.obs[0]]
        for step in range(batch.num_steps):
            next_s = self.wm.step(states[-1], enc_act[step]).sample()
            states.append(next_s)
        states = torch.stack(states)

        return states[1:-1].flatten(0, 1).detach()


class Actor:
    def __call__(self, s):
        ...


class Agent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor, device=None, num_envs=None):
        self.wm = wm
        self.actor = actor
        self._device = device

    def policy(self, obs):
        act = self.actor(obs.to(self._device)).sample()
        return self.wm.act_dec(act).cpu()
