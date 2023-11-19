from pathlib import Path

import torch
from tqdm.auto import tqdm

from rsrch.exp import comet
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron

from . import config, env
from .nets import *
from .utils import gae_adv_est


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)

    loader = env.Loader(cfg.env)

    val_envs = loader.val_envs(cfg.env_workers)
    train_envs = loader.val_envs(cfg.train_envs)

    ac = ActorCritic(
        obs_space=loader.obs_space,
        act_space=loader.act_space,
        share_enc=cfg.share_encoder,
        custom_init=cfg.custom_init,
    ).to(device)

    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.opt_eps)

    if cfg.alpha.autotune:
        if loader.discrete:
            assert isinstance(loader.act_space, gym.spaces.TensorDiscrete)
            max_ent = np.log(loader.act_space.n)
        else:
            assert isinstance(loader.act_space, gym.spaces.TensorBox)
            max_ent = torch.log(loader.act_space.high - loader.act_space.low).sum()
        target_ent = cfg.alpha.ent_scale * max_ent

        log_alpha = nn.Parameter(torch.zeros([]), requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr, eps=cfg.opt_eps)
        alpha = log_alpha.exp().item()
    else:
        alpha = cfg.alpha.value

    class ACAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = loader.conv_obs(obs).to(device)
            act_rv = ac(obs, values=False)
            return act_rv.sample().cpu().numpy()

    train_agent = ACAgent()
    val_agent = ACAgent()

    ep_ids = [None for _ in range(train_envs.num_envs)]
    buf = data.OnlineBuffer()
    env_iter = iter(rollout.steps(train_envs, train_agent))

    env_step = 0
    exp = comet.Experiment(project="ppo")
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.total_steps)

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_log = cron.Every(lambda: env_step, cfg.log_every)
    should_end = cron.Once(lambda: env_step >= cfg.total_steps)

    def val_epoch():
        val_returns = []
        val_iter = rollout.episodes(val_envs, val_agent, max_episodes=cfg.val_episodes)
        for _, ep in val_iter:
            val_returns.append(sum(ep.reward))
        exp.add_scalar("val/returns", np.mean(val_returns))

    def train_step():
        nonlocal alpha, env_step

        buf.reset()
        for _ in range(cfg.steps_per_epoch * cfg.train_envs):
            env_idx, step = next(env_iter)
            ep_ids[env_idx] = buf.push(ep_ids[env_idx], step)
            if "episode" in step.info:
                exp.add_scalar("train/ep_ret", step.info["episode"]["r"])
            env_step += 1
            pbar.update()

        train_eps = [*buf.values()]
        obs, act, logp, adv, ret, value = [], [], [], [], [], []

        with torch.no_grad():
            for ep in train_eps:
                ep_obs = loader.conv_obs(ep.obs).to(device)
                obs.append(ep_obs[:-1])
                ep_policy, ep_value = ac(ep_obs)
                if ep.term:
                    ep_value[-1] = 0.0
                value.append(ep_value[:-1])
                ep_reward = torch.tensor(ep.reward).type_as(ep_value)
                ep_adv, ep_ret = gae_adv_est(
                    ep_reward, ep_value, cfg.gamma, cfg.gae_lambda
                )
                adv.append(ep_adv)
                ret.append(ep_ret)
                ep_act = loader.conv_act(ep.act).to(device)
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
                v_loss = cfg.vf_coeff * v_loss

                policy_ent = new_policy.entropy().mean()
                ent_loss = alpha * -policy_ent

                loss = policy_loss + v_loss + ent_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(ac.parameters(), cfg.clip_grad)
                opt.step()

                if cfg.alpha.autotune:
                    alpha_loss = log_alpha * (policy_ent.detach() - target_ent)
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

                    if should_log:
                        exp.add_scalar("train/alpha", alpha)
                        exp.add_scalar("train/alpha_loss", alpha_loss)
                        exp.add_scalar("train/policy_ent", policy_ent)

        if should_log:
            exp.add_scalar("train/loss", loss)
            exp.add_scalar("train/policy_loss", policy_loss)
            exp.add_scalar("train/v_loss", v_loss)
            exp.add_scalar("train/ent_loss", ent_loss)
            exp.add_scalar("train/mean_v", value.mean())

    while True:
        if should_val:
            val_epoch()

        if should_end:
            break

        train_step()


if __name__ == "__main__":
    main()
