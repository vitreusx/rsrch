from functools import cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch.exp.pbar import ProgressBar
from rsrch.exp.wandb import Experiment
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron

from . import cem, config, env
from . import wm as wm_
from .utils import over_seq


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)
    env_f = env.make_factory(cfg.env, device)

    sampler = data.UniformSampler()
    buffer = env_f.chunk_buffer(cfg.buffer.capacity, cfg.seq_len, sampler)

    wm = wm_.WorldModel(env_f.obs_space, env_f.act_space)
    wm = wm.to(device)
    wm_opt = cfg.opt.make()(wm.parameters())

    venv = env_f.vector_env(1)

    actor = cem.CEMPlanner(cfg.cem, wm, env_f.act_space)
    opt_agent = env_f.VecAgent(wm_.VecAgent(wm, actor))
    rand_agent = gym.vector.agents.RandomAgent(venv)
    agent = gym.vector.agents.EpsAgent(opt_agent, rand_agent, eps=1.0)

    env_iter = iter(rollout.steps(venv, agent))
    ep_ids = [None] * venv.num_envs
    ep_rets = [0.0] * venv.num_envs

    exp = Experiment(project="pets_v2")
    board = exp.board
    env_step = 0
    board.add_step("env_step", lambda: env_step, default=True)
    should_log = cron.Every(lambda: env_step, 128)
    pbar = ProgressBar("PETS/v2", cfg.total_steps)

    while env_step < cfg.total_steps:
        for _ in range(cfg.env_steps):
            env_idx, step = next(env_iter)
            ep_ids[env_idx], _ = buffer.push(ep_ids[env_idx], step)
            ep_rets[env_idx] += step.reward
            if step.done:
                board.add_scalar("train/ep_ret", ep_rets[env_idx])
                ep_rets[env_idx] = 0
            env_step += 1
            pbar.step(1)
            agent.eps = 1.0 - env_step / cfg.total_steps

        if len(buffer) < cfg.buffer.prefill:
            continue

        for _ in range(cfg.opt_steps):
            idxes = sampler.sample(cfg.batch_size)
            batch = env_f.fetch_chunk_batch(buffer, idxes)

            obs = over_seq(wm.obs_enc)(batch.obs)
            act = over_seq(wm.act_enc)(batch.act)

            trans_h0 = wm.init_trans(obs[0])
            trans_x = torch.cat([act, obs[1:]], -1)
            trans_hx, _ = wm.trans(trans_x, trans_h0)
            trans_hx = torch.cat([trans_h0[[-1]], trans_hx], 0)

            L, N = act.shape[:2]
            idxes = torch.randint(0, L - 1, [N])
            pred_h0 = wm.init_pred(trans_hx[idxes, torch.arange(N)])
            pred_x = act[idxes, torch.arange(N)][None]
            pred_h1 = trans_hx[idxes + 1, torch.arange(N)]
            pred_hx, _ = wm.pred(pred_x, pred_h0)
            pred_hx = pred_hx[-1]

            hx_norm = over_seq(torch.linalg.norm)(trans_hx, dim=-1)
            norm_loss = F.mse_loss(hx_norm, torch.ones_like(hx_norm))

            pred_loss = F.mse_loss(pred_hx, pred_h1)

            term_rvs = wm.term(trans_hx[-1])
            term_loss = -term_rvs.log_prob(batch.term).mean()

            rew_rvs = over_seq(wm.reward)(trans_hx[1:])
            rew_loss = -rew_rvs.log_prob(batch.reward).mean()

            dec_rvs = over_seq(wm.dec)(trans_hx)
            dec_loss = -dec_rvs.log_prob(batch.obs).mean()

            loss = norm_loss + pred_loss + term_loss + rew_loss + dec_loss
            # loss = pred_loss + term_loss + rew_loss + dec_loss
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            wm_opt.step()

            if should_log:
                board.add_scalar(f"train/wm_loss", loss)
                for k in ["pred", "rew", "dec", "term", "norm"]:
                    board.add_scalar(f"train/{k}_loss", locals().get(f"{k}_loss"))


if __name__ == "__main__":
    main()
