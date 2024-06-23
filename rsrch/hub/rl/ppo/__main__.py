from collections import defaultdict
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
from tqdm.auto import tqdm

from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron, repro

from .. import env
from . import config
from .nets import *
from .utils import gae_adv_est


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")
    cfg = config.parse(cfg, config.Config)

    device = torch.device(cfg.device)

    repro.fix_seeds(seed=cfg.seed)

    env_f = env.make_factory(cfg.env, device, seed=cfg.seed)

    val_envs = env_f.vector_env(cfg.env_workers, mode="val")
    train_envs = env_f.vector_env(cfg.train_envs, mode="train")

    ac = ActorCritic(
        obs_space=env_f.obs_space,
        act_space=env_f.act_space,
        share_enc=cfg.share_encoder,
        custom_init=cfg.custom_init,
    ).to(device)

    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.opt_eps)

    class ACAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs).to(device)
            act_rv = ac(obs, values=False)
            return env_f.move_act(act_rv.sample(), to="env")

    train_agent = ACAgent()
    val_agent = ACAgent()

    buf = env_f.buffer(capacity=None)
    view = data.EpisodeView(buf)

    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)
    env_iter = iter(rollout.steps(train_envs, train_agent))

    exp = Experiment(
        project="ppo",
        run=f"{cfg.env.id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
        board=Tensorboard,
    )

    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)

    pbar = tqdm(total=cfg.total_steps)

    should_log = cron.Every(lambda: env_step, cfg.log_every)
    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_save = cron.Every(lambda: env_step, cfg.save_ckpt_every)

    def save_ckpt():
        ckpt_path = exp.dir / "ckpts" / f"env_step={env_step}.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "wb") as f:
            state = {"ac": ac.state_dict()}
            torch.save(state, f)

    def val_epoch():
        val_returns = []
        val_iter = rollout.episodes(val_envs, val_agent)
        val_iter = islice(val_iter, 0, cfg.val_episodes)
        for _, ep in val_iter:
            val_returns.append(sum(ep.reward))
        exp.add_scalar("val/returns", np.mean(val_returns))

    def train_step():
        buf.clear()
        ep_ids.clear()
        nonlocal env_step
        for _ in range(cfg.steps_per_epoch * cfg.train_envs):
            env_idx, step = next(env_iter)
            ep_ids[env_idx] = buf.push(ep_ids[env_idx], step)
            ep_rets[env_idx] += step.reward
            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                del ep_rets[env_idx]
            env_step += 1
            pbar.update()

        train_eps = [*view.values()]
        obs, act, logp, adv, ret, value = [], [], [], [], [], []

        with torch.no_grad():
            for ep in train_eps:
                ep_obs = env_f.move_obs(np.stack(ep.obs))
                obs.append(ep_obs[:-1])
                ep_policy, ep_value = ac(ep_obs)
                if ep.term:
                    ep_value[-1] = 0.0
                value.append(ep_value[:-1])
                ep_reward = torch.tensor(ep.reward).type_as(ep_value)
                ep_reward = ep_reward.sign()
                ep_adv, ep_ret = gae_adv_est(
                    ep_reward, ep_value, cfg.gamma, cfg.gae_lambda
                )
                adv.append(ep_adv)
                ret.append(ep_ret)
                ep_act = env_f.move_act(np.stack(ep.act))
                act.append(ep_act)
                ep_logp = ep_policy[:-1].log_prob(ep_act)
                logp.append(ep_logp)

        cat_ = [torch.cat(x) for x in (obs, act, logp, adv, ret, value)]
        obs, act, logp, adv, ret, value = cat_

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(act))
            for idxes in perm.split(cfg.update_batch):
                new_policy, new_value = ac(obs[idxes])
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
                    clipped_v = value[idxes] + (new_value - value[idxes]).clamp(
                        -cfg.clip_coeff, cfg.clip_coeff
                    )
                    v_loss1 = (new_value - ret[idxes]).square()
                    v_loss2 = (clipped_v - ret[idxes]).square()
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_value - ret[idxes]).square().mean()

                ent_loss = -new_policy.entropy().mean()

                loss = policy_loss + cfg.ent_coeff * ent_loss + cfg.vf_coeff * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(ac.parameters(), cfg.clip_grad)
                opt.step()

        if should_log:
            exp.add_scalar("train/loss", loss)
            exp.add_scalar("train/policy_loss", policy_loss)
            exp.add_scalar("train/v_loss", v_loss)
            exp.add_scalar("train/ent_loss", ent_loss)
            exp.add_scalar("train/mean_v", value.mean())

    while True:
        if should_save:
            save_ckpt()

        if should_val:
            val_epoch()

        if env_step >= cfg.total_steps:
            break

        train_step()


if __name__ == "__main__":
    main()
