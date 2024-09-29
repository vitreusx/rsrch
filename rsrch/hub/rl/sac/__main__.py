from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch import rl, spaces
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.nn import dh, fc
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import config, cron, data, repro

from .alpha import Alpha
from .config import Config
from .utils import to_camel_case


@dataclass
class Batch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device: torch.device):
        return Batch(
            obs=self.obs.to(device),
            act=self.act.to(device),
            next_obs=self.next_obs.to(device),
            reward=self.reward.to(device),
            term=self.term.to(device),
        )


def main():
    cfg = config.cli(
        config_yml=Path(__file__).parent / "config.yml",
    )
    cfg = config.cast(cfg, Config)

    repro.seed_all(cfg.random.seed)
    repro.set_fully_deterministic(cfg.random.deterministic)

    device = torch.device(cfg.device)
    compute_dtype = getattr(torch, cfg.compute_dtype)

    def autocast():
        enabled = compute_dtype != torch.float32
        return torch.autocast(device.type, compute_dtype, enabled)

    sdk = rl.sdk.make(cfg.env)
    assert isinstance(sdk.obs_space, spaces.torch.Box)
    assert isinstance(sdk.act_space, spaces.torch.Box)

    obs_dim = int(np.prod(sdk.obs_space.shape))
    act_dim = int(np.prod(sdk.act_space.shape))

    exp = Experiment(project="sac", prefix=sdk.id)
    exp.boards.append(Tensorboard(exp.dir / "board"))

    env_step, opt_step, agent_step = 0, 0, 0
    exp.register_step("env_step", lambda: env_step, default=True)
    exp.register_step("opt_step", lambda: opt_step)
    step_fns = {
        "env_step": (lambda: env_step),
        "opt_step": (lambda: opt_step),
        "agent_step": (lambda: agent_step),
    }

    def make_sched(cfg):
        cfg = {**cfg}
        nonlocal env_step, opt_step
        count, unit = cfg["every"], cfg["of"]
        del cfg["every"], cfg["of"]
        cfg = {**cfg, "step_fn": step_fns[unit], "period": count}
        return cron.Every(**cfg)

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self._fc = nn.Sequential(
                fc.FullyConnected(
                    layer_sizes=[obs_dim + act_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                nn.Flatten(0),
            )

        def forward(self, s: Tensor, a: Tensor):
            x = torch.cat([s.flatten(1), a.flatten(1)], -1)
            return self._fc(x)

    qf, qf_t = nn.ModuleList(), nn.ModuleList()
    for _ in range(2):
        qf.append(Q().to(device))
        qf_t.append(Q().to(device))

    def opt_ctor(spec: dict):
        spec = {**spec}
        cls = getattr(torch.optim, to_camel_case(spec["type"]))
        del spec["type"]
        return partial(cls, **spec)

    qf_opt = opt_ctor(cfg.value.opt)(qf.parameters())
    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(
        source=qf,
        target=qf_t,
        tau=cfg.value.polyak.tau,
        every=cfg.value.polyak.every,
    )

    class Actor(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(1),
                fc.FullyConnected(
                    layer_sizes=[obs_dim, cfg.hidden_dim, cfg.hidden_dim],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                dh.make(
                    layer_ctor=partial(nn.Linear, cfg.hidden_dim),
                    space=sdk.act_space,
                ),
            )

    actor = Actor().to(device)
    actor_opt = opt_ctor(cfg.actor.opt)(actor.parameters())

    class TrainAgent(gym.vector.agents.Markov):
        def __init__(self):
            super().__init__(sdk.obs_space, sdk.act_space)
            self.prefill = False

        def policy_from_last(self, obs: Tensor):
            if self.prefill:
                n = obs.shape[0]
                return self.act_space.sample(obs.shape[:1])
            else:
                actor.eval()
                with torch.inference_mode():
                    obs = obs.to(device)
                    with autocast():
                        policy: D.Distribution = actor(obs)
                        action = policy.sample()
                    return action.cpu()

    train_agent = TrainAgent()
    train_envs = sdk.make_envs(cfg.num_envs, mode="train")
    env_iter = iter(sdk.rollout(train_envs, train_agent))
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)
    env_total_steps = defaultdict(lambda: 0)

    should_opt = make_sched(cfg.value.sched)
    should_log_v = make_sched(cfg.log_sched)
    should_log_a = make_sched(cfg.log_sched)

    class ValAgent(gym.vector.agents.Markov):
        def __init__(self):
            super().__init__(sdk.obs_space, sdk.act_space)

        def policy_from_last(self, obs: Tensor):
            actor.eval()
            with torch.inference_mode():
                obs = obs.to(device)
                with autocast():
                    policy: D.Distribution = actor(obs)
                    action = policy.mode
                return action.cpu()

    val_agent = ValAgent()
    val_envs = sdk.make_envs(cfg.val.envs, mode="val")

    def make_val_iter():
        val_iter = sdk.rollout(val_envs, val_agent)
        val_iter = gym.utils.episodes(val_iter)
        val_iter = islice(val_iter, 0, cfg.val.episodes)
        return val_iter

    buf = sdk.wrap_buffer(rl.data.Buffer())
    buf = rl.data.SizeLimited(buf, cfg.buf_cap)

    buf = rl.data.Observable(buf)
    sampler = rl.data.Sampler()
    slices = rl.data.SliceView(2, sampler)
    buf.attach(slices)

    def make_train_iter():
        slice_iter = iter(slices)
        while True:
            batch = []
            for _ in range(cfg.batch_size):
                seq, pos = next(slice_iter)
                batch.append(seq)

            obs = torch.stack([seq[0]["obs"] for seq in batch])
            act = torch.stack([seq[1]["act"] for seq in batch])
            next_obs = torch.stack([seq[1]["obs"] for seq in batch])
            reward = torch.tensor([seq[1]["reward"] for seq in batch])
            term = torch.tensor([seq[1].get("term", False) for seq in batch])
            yield Batch(obs, act, next_obs, reward, term)

    train_iter = iter(make_train_iter())

    def take_env_step():
        nonlocal env_step, agent_step
        env_idx, (step, final) = next(env_iter)
        ep_ids[env_idx] = buf.push(ep_ids[env_idx], step, final)
        ep_rets[env_idx] += step.get("reward", 0.0)
        if "total_steps" in step:
            diff = step["total_steps"] - env_total_steps[env_idx]
            env_total_steps[env_idx] = step["total_steps"]
        else:
            diff = 1
        env_step += diff
        agent_step += 1
        if final:
            exp.add_scalar("train/ep_return", ep_rets[env_idx])
            del ep_ids[env_idx], ep_rets[env_idx]

    should_val = make_sched(cfg.val.sched)

    alpha = Alpha(cfg.alpha, sdk.act_space)

    pbar = tqdm(desc="Warmup", total=cfg.warmup, initial=env_step)

    train_agent.prefill = True
    while env_step <= cfg.warmup:
        take_env_step()
        pbar.update(env_step - pbar.n)

    train_agent.prefill = False
    pbar = tqdm(desc="SAC", total=cfg.total_steps, initial=env_step)
    while env_step <= cfg.total_steps:
        if should_val:
            val_iter = make_val_iter()
            val_iter = tqdm(val_iter, desc="Val", total=cfg.val.episodes, leave=False)

            val_rets = []
            for _, ep in val_iter:
                ep_ret = sum(step.get("reward", 0.0) for step in ep)
                val_rets.append(ep_ret)

            exp.add_scalar("val/mean_ret", np.mean(val_rets))

        actor.train()
        while should_opt:
            batch: Batch = next(train_iter)
            batch = batch.to(device)

            with torch.no_grad():
                with autocast():
                    next_act_rv = actor(batch.next_obs)
                    next_act = next_act_rv.sample()
                    min_q = torch.min(
                        qf_t[0](batch.next_obs, next_act),
                        qf_t[1](batch.next_obs, next_act),
                    )
                    next_v = min_q - alpha.value * next_act_rv.log_prob(next_act)
                    gamma = (1.0 - batch.term.float()) * cfg.gamma
                    q_targ = batch.reward + gamma * next_v

            with autocast():
                qf0_pred = qf[0](batch.obs, batch.act)
                qf1_pred = qf[1](batch.obs, batch.act)
                q_loss = F.mse_loss(qf0_pred, q_targ) + F.mse_loss(qf1_pred, q_targ)

            qf_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            qf_opt.step()
            qf_polyak.step()

            if opt_step % cfg.actor.opt_ratio == 0:
                for _ in range(cfg.actor.opt_ratio):
                    with autocast():
                        act_rv = actor(batch.obs)
                        act = act_rv.rsample()
                        min_q = torch.minimum(
                            qf[0](batch.obs, act),
                            qf[1](batch.obs, act),
                        )
                        actor_loss = -(
                            min_q - alpha.value * act_rv.log_prob(act)
                        ).mean()

                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()

                    if alpha.adaptive:
                        act_rv = actor(batch.obs)
                        mets = alpha.opt_step(act_rv.entropy())
                        for k, v in mets.items():
                            exp.add_scalar(f"train/alpha_{k}", v)

                    if should_log_a:
                        exp.add_scalar("train/actor_loss", actor_loss)

            if should_log_v:
                exp.add_scalar("train/mean_q", qf0_pred.mean())
                exp.add_scalar("train/q_loss", q_loss)

            opt_step += 1

        take_env_step()
        pbar.update(env_step - pbar.n)


if __name__ == "__main__":
    main()
