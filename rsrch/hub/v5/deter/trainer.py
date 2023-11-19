from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rsrch.exp.api import Experiment
from rsrch.rl import data
from rsrch.utils import cron

from ..common.utils import Optim, over_seq
from .wm import WorldModel


@dataclass
class Config:
    opt: Optim
    batch_size: int
    seq_len: int


class Context:
    should_log: cron.Flag
    exp: Experiment


class Trainer:
    def __init__(self, cfg: Config, wm: WorldModel, ctx: Context):
        self.cfg = cfg
        self.wm = wm
        self.wm_opt = cfg.opt.make()(self.wm.parameters())
        self.ctx = ctx

    def opt_step(self, batch: data.ChunkBatch):
        obs = over_seq(self.wm.obs_enc)(batch.obs)
        act = over_seq(self.wm.act_enc)(batch.act)

        trans_h0 = self.wm.init_trans(obs[0])
        trans_x = torch.cat([act, obs[1:]], -1)
        trans_hx, _ = self.wm.trans(trans_x, trans_h0)
        trans_hx = torch.cat([trans_h0[[-1]], trans_hx], 0)

        L, N = act.shape[:2]
        idxes = torch.randint(0, L - 1, [N])
        pred_h0 = self.wm.init_pred(trans_hx[idxes, torch.arange(N)])
        pred_x = act[idxes, torch.arange(N)][None]
        pred_h1 = trans_hx[idxes + 1, torch.arange(N)]
        pred_hx, _ = self.wm.pred(pred_x, pred_h0)
        pred_hx = pred_hx[-1]

        losses = []

        norm_loss = (trans_hx.square().sum(-1) - 1.0).square().mean()
        # hx_norm = over_seq(torch.linalg.norm)(trans_hx, dim=-1)
        # norm_loss = F.mse_loss(hx_norm, torch.ones_like(hx_norm))
        losses.append(norm_loss)

        pred_loss = F.mse_loss(pred_hx, pred_h1)
        losses.append(pred_loss)

        # pred_loss = 0.5 * (1.0 - F.cosine_similarity(pred_hx, pred_h1).mean())

        term_rvs = self.wm.term(trans_hx[-1])
        term_loss = -term_rvs.log_prob(batch.term).mean()
        losses.append(term_loss)

        rew_rvs = over_seq(self.wm.reward)(trans_hx[1:])

        rew_loss = -rew_rvs.log_prob(batch.reward).mean()
        losses.append(rew_loss)

        recon_rvs = over_seq(self.wm.recon)(trans_hx)
        recon_loss = -recon_rvs.log_prob(batch.obs).mean()
        losses.append(recon_loss)

        loss = sum(losses)
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.wm_opt.step()

        if self.ctx.should_log:
            exp = self.ctx.exp
            exp.add_scalar(f"train/wm_loss", loss)
            for k in ["pred", "rew", "recon", "term", "norm"]:
                if f"{k}_loss" not in locals():
                    continue
                exp.add_scalar(f"train/{k}_loss", locals()[f"{k}_loss"])

        return trans_hx.detach()
