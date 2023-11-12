from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym

from ..common.utils import over_seq
from .wm import WorldModel


@dataclass
class Config:
    niters: int
    pop: int
    elites: int
    horizon: int


class CEMPlanner:
    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        act_space: gym.TensorSpace,
    ):
        self.cfg = cfg
        self.wm = wm
        self.act_space = act_space
        self.act_dec = lambda x: x

    @torch.inference_mode()
    def __call__(self, s: Tensor):
        # state: [N, H]

        # Initialize action sequence distribution
        batch_size = s.shape[0]
        if isinstance(self.act_space, gym.spaces.TensorBox):
            shape = [self.cfg.horizon, batch_size, *self.act_space.shape]
            act_shape = [*self.act_space.shape]
            loc = torch.zeros(shape, dtype=s.dtype, device=s.device)
            scale = torch.ones(shape, dtype=s.dtype, device=s.device)
            act_seq_rv = D.Normal(loc, scale, len(self.act_space.shape))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            shape = [self.cfg.horizon, batch_size, self.act_space.n]
            act_shape = []
            probs = torch.ones(shape, dtype=s.dtype, device=s.device)
            probs = probs / self.act_space.n
            act_seq_rv = D.Categorical(probs=probs)

        init_s = s.repeat(self.cfg.pop, 1)  # [Pop * N, H]

        for _ in range(self.cfg.niters):
            # Sample action sequences
            acts = act_seq_rv.sample([self.cfg.pop])  # [Pop, L, N, *A]

            # Use prediction RNN to get predicted states.
            pred_x = acts.swapaxes(0, 1).flatten(1, 2)  # [L, Pop * N, *A]
            pred_x = over_seq(self.wm.act_enc)(pred_x)  # [L, Pop * N, D_A]

            all_s, cur_s = [], init_s
            for step in range(self.cfg.horizon):
                next_s = self.wm.pred(pred_x[step], cur_s).sample()
                all_s.append(next_s)
                cur_s = next_s
            all_s = torch.stack(all_s)  # [L, Pop * N, H]

            # Compute returns for the trajectories
            term_rvs = over_seq(self.wm.term)(all_s)
            term = term_rvs.mean
            cont = (1 - term).cumprod(0)
            rew_rvs = over_seq(self.wm.reward)(all_s)
            rew = rew_rvs.mean
            ret = (cont * rew).sum(0).reshape(self.cfg.pop, batch_size)

            idxes = torch.topk(ret, self.cfg.elites, dim=0, sorted=False).indices
            # idxes: [Elites, N]
            shape_r = [self.cfg.elites, 1, batch_size, *(1 for _ in act_shape)]
            shape_e = [self.cfg.elites, acts.shape[1], batch_size, *acts.shape[3:]]
            idxes = idxes.reshape(shape_r).expand(shape_e)
            elites = acts.gather(0, idxes)  # [Elites, L, N, *A]

            if isinstance(self.act_space, gym.spaces.TensorBox):
                loc, scale = elites.mean(0), elites.std(0)
                act_seq_rv = D.Normal(loc, scale, len(self.act_space.shape))
            elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
                probs = F.one_hot(elites.long(), self.act_space.n).float().mean(0)
                act_seq_rv = D.Categorical(probs=probs)

        return D.Dirac(act_seq_rv.mode[0], len(act_shape))
