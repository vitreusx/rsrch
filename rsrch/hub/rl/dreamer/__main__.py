from collections import defaultdict
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp.tensorboard import Experiment
from rsrch.nn.utils import infer_ctx
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.utils import cron

from .. import env
from ..utils import over_seq
from . import config, nets, rssm


class OneHotEncoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.one_hot(input, self.num_classes)


def main():
    cfg_d = config.cli(
        config_file=Path(__file__).parent / "config.yml",
    )
    cfg = config.from_dicts([cfg_d], config.Config)

    exp = Experiment(
        project="dreamer",
        run=f"{cfg.env.id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
        config=cfg_d,
    )

    device = torch.device(cfg.device)
    env_f = env.make_factory(cfg.env, device=device, seed=cfg.random.seed)

    if isinstance(env_f.obs_space, spaces.torch.Image):
        obs_enc = nets.VisEncoder(
            env_f.obs_space,
            conv_hidden=32,
        )
    elif isinstance(env_f.obs_space, spaces.torch.Box):
        obs_enc = nn.Flatten()

    if isinstance(env_f.act_space, spaces.torch.Box):
        act_enc = nn.Flatten()
    elif isinstance(env_f.act_space, spaces.torch.Discrete):
        act_enc = OneHotEncoder(env_f.act_space.n)

    with infer_ctx(obs_enc, act_enc):
        test_obs = env_f.obs_space.sample([1])
        obs_dim = obs_enc(test_obs).shape[1]
        test_act = env_f.act_space.sample([1])
        act_dim = act_enc(test_act).shape[1]

    wm: rssm.RSSM
    wm_opt: torch.optim.Optimizer

    state_dim = wm.state_dim

    if isinstance(env_f.obs_space, spaces.torch.Image):
        obs_pred = nn.Sequential(
            rssm.StateToTensor(),
            nets.VisDecoder(
                env_f.obs_space,
                state_dim=state_dim,
                conv_hidden=32,
            ),
        )
    elif isinstance(env_f.obs_space, spaces.torch.Box):
        obs_pred = nn.Sequential(
            rssm.StateToTensor(),
            nn.Linear(state_dim, int(np.prod(env_f.obs_space.shape))),
            nets.Reshape(env_f.obs_space.shape),
        )

    env_buf: data.SliceBuffer

    train_envs: gym.VectorEnv
    seq_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    val_envs: gym.VectorEnv
    val_agent: gym.VecAgent
    make_val_iter = lambda: iter(rollout.episodes(val_envs, val_agent))

    env_step, true_step, opt_step = 0, 0, 0
    get_step = {k: lambda: locals()[k] for k in locals() if k.endswith("_step")}

    exp.register_step("true_step", lambda: true_step, default=True)
    exp.register_step("env_step", lambda: env_step)
    exp.register_step("opt_step", lambda: opt_step)

    def make_until(cfg):
        value, unit = cfg
        step_fn = lambda: get_step(unit)
        return cron.Until(step_fn, value)

    should_warmup: cron.Until
    should_train: cron.Until

    def make_every(cfg):
        value, unit = cfg["every"]
        step_fn = lambda: get_step(unit)
        cfg = {**cfg, "every": value, "step_fn": step_fn}
        return cron.Every4(**cfg)

    should_val: cron.Every4
    should_opt_policy: cron.Every4
    should_opt_wm: cron.Every4
    should_log: cron.Every4

    warmup_agent = gym.vector.agents.RandomAgent(train_envs)
    warmup_iter = iter(rollout.steps(train_envs, warmup_agent))
    seq_ids.clear()
    ep_rets.clear()

    def take_env_step(env_iter):
        env_idx, step = next(env_iter)
        seq_ids[env_idx], _ = env_buf.push(seq_ids[env_idx], step)
        ep_rets[env_idx] += step.reward

        if step.done:
            exp.add_scalar("train/ep_ret", ep_rets[env_idx])
            del seq_ids[env_idx]
            del ep_rets[env_idx]

        nonlocal env_step, true_step
        env_step += 1
        true_step += getattr(env_f, "frame_skip", 1)

    while should_warmup:
        take_env_step(warmup_iter)

    train_agent: gym.VecAgent
    train_iter = iter(rollout.steps(train_envs, train_agent))

    while True:
        if should_val:
            val_iter = islice(make_val_iter(), 0, cfg.val_episodes)
            val_ep_rets = [sum(ep.reward) for ep in val_iter]
            exp.add_scalar("val/mean_ep_ret", np.mean(val_ep_rets))

        if not should_train:
            break

        take_env_step(train_iter)

        while should_opt_wm:
            idxes = env_buf.sampler.sample(cfg.batch_size)
            batch = env_f.fetch_slice_batch(env_buf, idxes)

            enc_obs = over_seq(obs_enc)(batch.obs)
            enc_act = over_seq(act_enc)(batch.act)

            # p(\beta_t | o_1... o_{t-1}, a_{t-1}, o_t)
            beliefs = wm.beliefs(enc_act, enc_obs)
            # z_t ~ p(z_t | \beta_t)
            prior_dist = over_seq(wm.proj)(beliefs)
            z_samp = prior_dist.rsample()
            # p(z_t | z_{t-1}, a_{t-1})
            pred_dist = over_seq(wm.pred)(z_samp[:-1], enc_act)

            losses = {}

            div_loss1 = rssm.state_dist_div(pred_dist.detach(), prior_dist)
            div_loss2 = rssm.state_dist_div(pred_dist, prior_dist.detach())
            losses["div"] = cfg.wm.alpha * div_loss1 + (1.0 - cfg.wm.alpha) * div_loss2

            obs_dist = over_seq(obs_pred)(z_samp)
            losses["obs"] = -obs_dist.log_prob(batch.obs).mean()
            rew_dist = over_seq(rew_pred)(z_samp)
            losses["rew"] = -rew_dist.log_prob(batch.reward).mean()
            cont_dist = over_seq(cont_pred)(z_samp)
            losses["cont"] = -cont_dist.log_prob(1.0 - batch.term).mean()

            wm_loss = sum(cfg.wm.coef[k] * losses[k] for k in losses)

            wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            wm_opt.step()

            if should_log:
                for k, v in losses:
                    exp.add_scalar(f"train/{k}_loss", v)
                exp.add_scalar(f"train/wm_loss", wm_loss)

        while should_opt_policy:
            ...


if __name__ == "__main__":
    main()
