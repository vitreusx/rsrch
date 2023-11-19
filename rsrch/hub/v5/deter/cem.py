from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym

from ..agent.cem import Config
from ..common.utils import over_seq
from .wm import WorldModel


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

    @torch.inference_mode()
    def __call__(self, h: Tensor):
        # h: [#Layers_T, N, H]

        batch_size = h.shape[1]
        if isinstance(self.act_space, gym.spaces.TensorBox):
            shape = [self.cfg.horizon, batch_size, *self.act_space.shape]
            act_shape = [*self.act_space.shape]
            loc = torch.zeros(shape, dtype=h.dtype, device=h.device)
            scale = torch.ones(shape, dtype=h.dtype, device=h.device)
            act_seq_rv = D.Normal(loc, scale, len(self.act_space.shape))
        elif isinstance(self.act_space, gym.spaces.TensorDiscrete):
            shape = [self.cfg.horizon, batch_size, self.act_space.n]
            act_shape = []
            probs = torch.empty(shape, dtype=h.dtype, device=h.device)
            probs.fill_(1.0 / self.act_space.n)
            act_seq_rv = D.Categorical(probs=probs)

        pred_h0 = self.wm.init_pred(h[-1])  # [#Layers_P, N, H]
        pred_h0 = pred_h0.repeat(1, self.cfg.pop, 1)  # [#Layers_P, Pop * N, H]

        for _ in range(self.cfg.niters):
            acts = act_seq_rv.sample([self.cfg.pop])  # [Pop, L, N, *A]

            pred_x = acts.swapaxes(0, 1).flatten(1, 2)  # [L, Pop * N, *A]
            pred_x = over_seq(self.wm.act_enc)(pred_x)  # [L, Pop * N, D_A]
            pred_hx, _ = self.wm.pred(pred_x, pred_h0)

            term_rvs = over_seq(self.wm.term)(pred_hx)
            term = term_rvs.mean
            cont = (1 - term).cumprod(0)
            rew_rvs = over_seq(self.wm.reward)(pred_hx)
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

        return act_seq_rv.mode[0]
