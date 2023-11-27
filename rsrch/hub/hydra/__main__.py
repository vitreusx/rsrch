from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ruamel import yaml
from torch import Tensor, nn

from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.nn import dist_head as dh
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import config, cron

from . import alpha, env, utils
from .utils import gae_adv_est


@dataclass
class Config:
    env: env.Config
    alpha: alpha.Config
    device: Literal["cuda", "cpu"] = "cuda"
    env_batch: int = 1024
    num_envs: int = 8
    opt_iters: int = 4
    opt_batch: int = 128
    prefill: int = int(10e3)
    clip_coeff: float = 0.2
    clip_vloss: bool = True
    adv_norm: bool = True
    vf_coeff: float = 0.5
    clip_grad: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95


def main():
    def_cfg_path = Path(__file__).parent / "config.yml"
    with open(def_cfg_path, "r") as def_cfg_f:
        def_cfg = yaml.safe_load(def_cfg_f)

    preset = "CartPole-v1"
    presets_path = Path(__file__).parent / "presets.yml"
    with open(presets_path, "r") as presets_f:
        presets = yaml.safe_load(presets_f)
        preset_cfg = presets.get(preset, {})

    cfg = config.from_dicts([def_cfg, preset_cfg], Config)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available:
        device = "cpu"
    device = torch.device(device)

    env_f = env.make_factory(cfg.env, device)
    assert isinstance(env_f.obs_space, gym.spaces.TensorBox)
    obs_dim = int(np.prod(env_f.obs_space.shape))
    assert isinstance(env_f.act_space, gym.spaces.TensorDiscrete)
    act_dim = env_f.act_space.n

    def make_actor():
        def layer_init(layer):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        head = dh.OneHotCategoricalST(128, act_dim, min_pr=1e-8)
        head.apply(layer_init)

        return nn.Sequential(
            nn.Flatten(1),
            # nn.BatchNorm1d(obs_dim),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            head,
        ).to(device)

    act_enc = lambda x: nn.functional.one_hot(x.long(), act_dim).float()
    act_dec = lambda x: x.argmax(-1)

    actor = make_actor()
    # actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)

    def make_critic():
        return nn.Sequential(
            nn.Flatten(1),
            # nn.BatchNorm1d(obs_dim),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Flatten(0),
        ).to(device)

    critic = make_critic()
    # critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)

    ac_params = [*actor.parameters(), *critic.parameters()]
    ac_opt = torch.optim.Adam(ac_params, lr=3e-4, eps=1e-5)

    class make_agent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            actor.eval()
            act_rv = actor(env_f.move_obs(obs))
            actor.train()
            act = act_dec(act_rv.sample())
            return env_f.move_act(act, to="env")

    envs = env_f.vector_env(num_envs=cfg.num_envs)
    agent = make_agent()
    env_iter = iter(rollout.steps(envs, agent))

    sampler = data.UniformSampler()
    q_buf = env_f.step_buffer(int(100e3), sampler)
    q_ep_ids = [None for _ in range(envs.num_envs)]

    def make_q():
        return nn.Sequential(
            nn.Flatten(1),
            # nn.BatchNorm1d(obs_dim),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        ).to(device)

    q1, q1_t = make_q(), make_q()
    polyak.sync(q1, q1_t)
    q1_polyak = polyak.Polyak(q1, q1_t, tau=0.995)

    q2, q2_t = make_q(), make_q()
    polyak.sync(q2, q2_t)
    q2_polyak = polyak.Polyak(q2, q2_t, tau=0.995)

    q_params = [*q1.parameters(), *q2.parameters()]
    q_opt = torch.optim.Adam(q_params, lr=3e-4, eps=1e-5)

    alpha_ = alpha.Alpha(cfg.alpha, env_f.act_space)

    exp = tensorboard.Experiment(project="hydra")
    env_step, total_steps = 0, int(100e3)
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(total=total_steps)

    should_log = cron.Every(lambda: env_step, 128)
    should_stop = cron.Once(lambda: env_step >= total_steps)

    def incr_env_step(n=1):
        nonlocal env_step
        env_step += n
        pbar.update(n)
        q1_polyak.step(n)
        q2_polyak.step(n)

    vpg_buf = data.OnlineBuffer()
    vpg_ep_ids = [None for _ in range(envs.num_envs)]

    ep_rets = [0.0 for _ in range(envs.num_envs)]

    while not should_stop:
        for _ in range(cfg.env_batch):
            env_idx, step = next(env_iter)
            q_ep_ids[env_idx], _ = q_buf.push(q_ep_ids[env_idx], step)
            vpg_ep_ids[env_idx] = vpg_buf.push(vpg_ep_ids[env_idx], step)

            ep_rets[env_idx] += step.reward
            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                ep_rets[env_idx] = 0.0

            incr_env_step()

        if env_step < cfg.prefill:
            continue

        cur_eps = [*vpg_buf.values()]
        obs, act, logp, adv, ret, val = [], [], [], [], [], []

        with torch.no_grad():
            for ep in cur_eps:
                ep_obs = env_f.move_obs(np.stack(ep.obs))
                obs.append(ep_obs[:-1])
                ep_policy = actor(ep_obs)
                ep_value = critic(ep_obs)
                if ep.term:
                    ep_value[-1] = 0.0
                val.append(ep_value[:-1])
                ep_reward = torch.tensor(ep.reward).type_as(ep_value)
                ep_adv, ep_ret = gae_adv_est(
                    ep_reward, ep_value, cfg.gamma, cfg.gae_lambda
                )
                adv.append(ep_adv)
                ret.append(ep_ret)
                ep_act = act_enc(env_f.move_act(np.stack(ep.act)))
                act.append(ep_act)
                ep_logp = ep_policy[:-1].log_prob(ep_act)
                logp.append(ep_logp)

        obs, act, logp, adv, ret, val = [
            torch.cat(x) for x in (obs, act, logp, adv, ret, val)
        ]

        for _ in range(cfg.opt_iters):
            perm = torch.randperm(len(act))
            for idxes in perm.split(cfg.opt_batch):
                new_policy = actor(obs[idxes])
                new_value = critic(obs[idxes])
                new_logp = new_policy.log_prob(act[idxes])
                log_ratio = new_logp - logp[idxes]
                ratio = log_ratio.exp()

                adv_ = adv[idxes]
                if cfg.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)

                t1 = -adv_ * ratio
                t2 = -adv_ * ratio.clamp(1 - cfg.clip_coeff, 1 + cfg.clip_coeff)
                policy_loss = torch.max(t1, t2).mean()

                if cfg.clip_vloss:
                    clipped_v = val[idxes] + (new_value - val[idxes]).clamp(
                        -cfg.clip_coeff, cfg.clip_coeff
                    )
                    v_loss1 = (new_value - ret[idxes]).square()
                    v_loss2 = (clipped_v - ret[idxes]).square()
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_value - ret[idxes]).square().mean()
                v_loss = cfg.vf_coeff * v_loss

                policy_ent = new_policy.entropy()
                ent_loss = alpha_.value * -policy_ent.mean()

                loss = policy_loss + v_loss + ent_loss
                ac_opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(ac_params, cfg.clip_grad)
                ac_opt.step()

                _, alpha_mets = alpha_.opt_step(policy_ent, return_metrics=True)

        if should_log:
            exp.add_scalar("train/loss", loss)
            exp.add_scalar("train/policy_loss", policy_loss)
            exp.add_scalar("train/v_loss", v_loss)
            exp.add_scalar("train/ent_loss", ent_loss)
            exp.add_scalar("train/mean_v", val.mean())
            for name, value in alpha_mets.items():
                exp.add_scalar(name, value)


if __name__ == "__main__":
    main()
