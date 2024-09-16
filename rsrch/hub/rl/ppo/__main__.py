from collections import defaultdict
from datetime import datetime
from functools import cached_property
from itertools import islice
from pathlib import Path

import torch
from tqdm.auto import tqdm

from rsrch import rl
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.rl import gym
from rsrch.rl.types import Slices
from rsrch.utils import cron, repro

from . import config
from .nets import *
from .utils import gae_adv_est


class ScaledOptimizer:
    def __init__(self, opt: torch.optim.Optimizer):
        self.opt = opt

    @cached_property
    def parameters(self) -> list[nn.Parameter]:
        params = []
        for group in self.opt.param_groups:
            params.extend(group["params"])
        return params

    @cached_property
    def device(self) -> torch.device:
        return self.parameters[0].device

    @cached_property
    def scaler(self) -> torch.cuda.amp.GradScaler:
        return getattr(torch, self.device.type).amp.GradScaler()

    def step(self, loss: Tensor, clip_grad: float | None = None):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()

    def state_dict(self):
        return {
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])


def main():
    cfg = config.open(Path(__file__).parent / "config.yml")
    cfg = config.cast(cfg, config.Config)

    device = torch.device(cfg.device)
    compute_dtype = getattr(torch, cfg.compute_dtype)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=compute_dtype)

    repro.seed_all(seed=cfg.seed)

    sdk = rl.sdk.make(cfg.env)

    val_envs = sdk.make_envs(cfg.env_workers, mode="val")
    train_envs = sdk.make_envs(cfg.train_envs, mode="train")

    ac = ActorCritic(
        obs_space=sdk.obs_space,
        act_space=sdk.act_space,
        share_enc=cfg.share_encoder,
        custom_init=cfg.custom_init,
    )
    ac = ac.to(device)

    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=cfg.opt_eps)
    opt = ScaledOptimizer(opt)

    class ACAgent(gym.agents.Memoryless):
        @torch.inference_mode()
        def _policy(self, obs):
            with autocast():
                act_rv = ac(obs.to(device), values=False)
                return act_rv.sample().cpu()

    train_agent = ACAgent()
    val_agent = ACAgent()

    buf = sdk.wrap_buffer(rl.data.Buffer())

    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)
    env_iter = iter(sdk.rollout(train_envs, train_agent))

    dt = datetime.now()
    date, time = f"{dt:%Y-%m-%d}", f"{dt:%H-%M-%S}"

    exp = Experiment(
        project="ppo",
        run_dir=f"runs/ppo/{date}/{sdk.id}__{time}",
    )
    exp.boards.append(Tensorboard(exp.dir / "board"))

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
        val_returns, ep_returns = [], defaultdict(lambda: 0.0)
        val_iter = sdk.rollout(val_envs, val_agent)
        while True:
            env_idx, (step, final) = next(val_iter)
            ep_returns[env_idx] += step.get("reward", 0.0)
            if final:
                val_returns.append(ep_returns[env_idx])
                del ep_returns[env_idx]
                if len(val_returns) >= cfg.val_episodes:
                    break

        exp.add_scalar("val/returns", np.mean(val_returns))

    def train_step():
        buf.clear()
        ep_ids.clear()
        nonlocal env_step
        for _ in range(cfg.steps_per_epoch * cfg.train_envs):
            env_idx, (step, final) = next(env_iter)
            ep_ids[env_idx] = buf.push(ep_ids[env_idx], step, final)
            ep_rets[env_idx] += step.get("reward", 0.0)
            if final:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                del ep_ids[env_idx], ep_rets[env_idx]
            env_step += 1
            pbar.update()

        train_eps: list[Slices] = []
        for seq in buf.values():
            if len(seq) < 8:
                continue
            seq = Slices(
                obs=torch.stack([step["obs"] for step in seq]),
                act=torch.stack([step["act"] for step in seq[1:]]),
                reward=torch.tensor(np.array([step["reward"] for step in seq[1:]])),
                term=torch.tensor(np.array([step.get("term", False) for step in seq])),
            )
            seq = seq.to(device)
            train_eps.append(seq)

        obs, act, logp, adv, ret, value = [], [], [], [], [], []

        with torch.no_grad():
            with autocast():
                for ep in train_eps:
                    obs.append(ep.obs[:-1])
                    ep_policy, ep_value = ac(ep.obs)
                    cont = 1.0 - ep.term.float()
                    ep_reward = ep.reward.sign()
                    ep_value, ep_reward = ep_value * cont, ep_reward * cont[:-1]
                    value.append(ep_value[:-1])
                    ep_adv, ep_ret = gae_adv_est(
                        ep_reward, ep_value, cfg.gamma, cfg.gae_lambda
                    )
                    adv.append(ep_adv)
                    ret.append(ep_ret)
                    act.append(ep.act)
                    ep_logp = ep_policy[:-1].log_prob(ep.act)
                    logp.append(ep_logp)

        cat_ = [torch.cat(x) for x in (obs, act, logp, adv, ret, value)]
        obs, act, logp, adv, ret, value = cat_

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(act))
            for idxes in perm.split(cfg.update_batch):
                with autocast():
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

                    loss = (
                        policy_loss + cfg.ent_coeff * ent_loss + cfg.vf_coeff * v_loss
                    )

                opt.step(loss, cfg.clip_grad)

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
