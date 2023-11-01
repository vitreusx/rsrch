from functools import cache

import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch.rl import data

from . import wm
from .config import Config


@cache
def over_seq(_func):
    """Transform a function that operates on batches (B, ...) to operate on
    sequences (L, B, ...)."""

    def _lifted(x: Tensor, *args, **kwargs):
        batch_size = x.shape[1]
        y = _func(x.flatten(0, 1), *args, **kwargs)
        return y.reshape(-1, batch_size, *y.shape[1:])

    return _lifted


class Trainer:
    def __init__(self, cfg: Config, wm: wm.WorldModel):
        self.cfg = cfg
        self.wm = wm
        self.wm_opt = self.cfg.opt.make()(self.wm.parameters())

    def opt_step(self, batch: data.ChunkBatch, ctx):
        obs = over_seq(self.wm.obs_enc)(batch.obs)
        act = over_seq(self.wm.act_enc)(batch.act)

        trans_h0 = self.wm.init(obs[0])
        trans_h0 = trans_h0[None].repeat(self.wm.trans.num_layers, 1, 1)
        trans_x = torch.cat([act, obs[1:]], -1)
        trans_hx, _ = self.wm.trans(trans_x, trans_h0)
        trans_hx = torch.cat([trans_h0[[-1]], trans_hx], 0)

        L, N = act.shape[:2]
        idxes = torch.arange(0, L - 1, [N])
        pred_h0 = trans_hx[idxes, torch.arange(N)]
        pred_x = act[idxes, torch.arange(N)][None]
        pred_h1 = trans_hx[idxes + 1, torch.arange(N)]
        pred_hx, _ = self.wm.pred(pred_x, pred_h0)
        pred_hx = pred_hx[-1]

        hx_norm = over_seq(torch.linalg.norm)(trans_hx, dim=-1)
        norm_loss = F.mse_loss(hx_norm, torch.ones_like(hx_norm))

        pred_loss = F.mse_loss(pred_hx, pred_h1)

        term_rvs = self.wm.term(trans_hx[-1])
        term_loss = -term_rvs.log_prob(batch.term).mean()

        rew_rvs = over_seq(self.wm.reward)(trans_hx[1:])
        rew_loss = -rew_rvs.log_prob(batch.reward).mean()

        dec_rvs = over_seq(self.wm.dec)(trans_hx)
        dec_loss = -dec_rvs.log_prob(batch.obs).mean()

        loss = norm_loss + pred_loss + term_loss + rew_loss + dec_loss
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.wm_opt.step()

        if ctx.should_log:
            ctx.board.add_scalar(f"train/wm_loss", loss)
            for k in ["norm", "pred", "rew", "dec"]:
                var = f"{k}_loss"
                ctx.board.add_scalar(var, locals().get(var))
