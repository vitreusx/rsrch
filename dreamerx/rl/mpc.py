from dataclasses import dataclass

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from dreamerx.common.utils import null_ctx
from rsrch import spaces


@dataclass
class Config:
    lookahead: int
    num_samples: int
    num_elites: int
    num_iters: int


class Actor:
    def __init__(
        self,
        cfg: Config,
        wm: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        self.cfg = cfg
        self.wm = wm
        self.device = next(self.wm.parameters()).device
        self.act_space: spaces.torch.Tensor = self.wm.act_space
        if not isinstance(self.act_space, (spaces.torch.OneHot,)):
            raise ValueError(self.act_space)
        self.compute_dtype = compute_dtype

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
            )

    def __call__(self, cur_state: Tensor):
        batch_size = states.shape[0]
        act_shape = (self.cfg.lookahead, batch_size, *self.act_space.shape)

        kw = dict(dtype=cur_state.dtype, device=cur_state.device)
        if isinstance(self.act_space, spaces.torch.OneHot):
            act_dist = D.OneHot(logits=torch.zeros(act_shape, **kw))

        for _ in range(self.cfg.num_iters):
            init_state = cur_state[None].expand(self.cfg.num_samples, *cur_state.shape)
            states = [init_state]
            acts = act_dist.sample((self.cfg.num_samples,))  # [S, L, N, *A]
            for _ in range(self.cfg.lookahead):
                next_state = self.wm.img_step(states[-1])
