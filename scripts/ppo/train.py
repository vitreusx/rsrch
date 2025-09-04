from collections import defaultdict
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Callable, Literal, TypedDict

import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from ruamel.yaml import YAML
from torch import Tensor, nn
from torch.optim import Optimizer

import rsrch.torch.distributions as D
from rsrch import rl, spaces
from rsrch.exp import Experiment, boards
from rsrch.rl import gym, sdk
from rsrch.rl.data import Buffer
from rsrch.rl.loaders import OnPolicyRLLoader
from rsrch.rl.utils import polyak
from rsrch.rl.utils.gae import gen_adv_est
from rsrch.torch.nn import dh
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.torch.nn.utils import shape_infer_mode
from rsrch.utils import cron, repro
from rsrch.utils.cast import cast

# isort: off
from config import Config
# isort: on


def layer_init(layer: nn.Linear | nn.Conv2d, std=np.sqrt(2), bias_const=0.0):
    """Do a custom init of a layer.

    Taken from cleanrl."""

    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariEncoder(nn.Sequential):
    """A standard conv encoder for Atari frames."""

    def __init__(self, obs_space: spaces.torch.Image):
        assert obs_space.shape[1:] == (84, 84)
        super().__init__(
            layer_init(nn.Conv2d(obs_space.num_channels, 32, 8, 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )


def ActorHead(in_features: int, act_space: spaces.torch.Discrete):
    def layer_ctor(out_features: int):
        return layer_init(nn.Linear(in_features, out_features), std=1e-2)

    return dh.Categorical(layer_ctor, act_space)


class Actor(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space

        self.encoder = AtariEncoder(obs_space)

        test_obs = obs_space.sample((1,))
        with shape_infer_mode(self.encoder):
            self.num_features = self.encoder(test_obs).shape[-1]

        self.head = ActorHead(self.num_features, act_space)

    def forward(self, obs: Tensor) -> D.Categorical:
        return self.head(self.encoder(obs))


class CriticHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.layer = layer_init(nn.Linear(in_features, 1), std=1.0)

    def forward(self, features: Tensor) -> Tensor:
        values: Tensor = self.layer(features)
        return values.ravel()


class Critic(nn.Module):
    def __init__(self, obs_space: spaces.torch.Image):
        super().__init__()
        self.obs_space = obs_space

        self.encoder = AtariEncoder(obs_space)

        test_obs = obs_space.sample((1,))
        with shape_infer_mode(self.encoder):
            in_features = self.encoder(test_obs).shape[-1]

        self.head = CriticHead(in_features)

    def forward(self, obs: Tensor) -> Tensor:
        return self.head(self.encoder(obs))


class RLSlices(TypedDict):
    obs: Tensor
    """A sequence of observations :math:`o_{1:L+1}` of shape :math:`(L+1, *O)`."""
    act: Tensor
    """A sequence of actions :math:`a_{1:L}` of shape :math:`(L, *A)`."""
    reward: Tensor
    """A sequence of rewards :math:`r_{2:L+1}` of shape :math:`(L)`, obtained
    upon arriving at :math:`o_{2:L+1}` (note that the first obs is skipped.)"""
    term: Tensor
    r"""A sequence of term values :math:`\tau_{0:L+1}` of shape :math:`(L+1)`.
    The values can be either boolean or float, to represent "fuzzy states"."""


@dataclass
class PPOData:
    """Data for PPO. The format is a flat list of observations, actions etc.

    NOTE: The sequence(s) don't need to correspond to MDP sequences."""

    obs: Tensor
    """A set of observations :math:`o_i`."""
    act: Tensor
    """A set of actions :math:`a_i` performed in state `i`."""
    logp: Tensor
    r"""The values of action log-probabilities :math:`\log{\pi(a_i \mid o_i)}`."""
    adv: Tensor
    """The advantage values :math:`A_i`."""
    ret: Tensor
    """The returns, or return estimates :math:`G_i`."""
    val: Tensor
    """The value estimates :math:`V_i`."""
    weight: Tensor | None
    """Item weights. The loss values are multiplied by it. Can be used for e.g.
    prioritization or masking."""


class ACForward:
    def __call__(
        self,
        actor: Actor,
        critics: tuple[Critic, ...],
        obs: Tensor,
    ) -> tuple[D.Distribution, tuple[Tensor, ...]]:
        """Compute policy and values for an actor and a set of critics.

        The rationale behind introducing such a construct is that some
        implementations share layers between the actor and the critic(s);
        supporting both separate and shared setups is best done by providing
        a single function to handle both cases uniformly."""


def get_minibatches(batch_size: int, num_mb: int):
    """Divide a batch into a number of minibatches.

    The minibatch sizes are selected in such a way, that they are divisible by 32,
    except for the last one. The last batch may be larger than the previous ones.
    """

    WARP = 32
    batch_size_w = batch_size // WARP
    mb_size = WARP * (batch_size_w // num_mb)
    mb_size_rem = batch_size - num_mb * mb_size
    split_sizes = [mb_size] * num_mb
    split_sizes[-1] += mb_size_rem
    end = np.cumsum(split_sizes)
    start = end - np.array(split_sizes)
    return [slice(start_i, end_i) for start_i, end_i in zip(start, end)]


get_minibatches = cache(get_minibatches)


class PPO:
    """A trainer class for PPO."""

    def __init__(
        self,
        *,
        actor: Actor,
        make_critic: Callable[[], Critic],
        ac_forward: ACForward | None,
        make_opt: Callable[[list[nn.Parameter]], Optimizer],
        update_epochs: int,
        mb_size: int | None,
        adv_norm: bool,
        clip_coef: float,
        clip_vloss: bool,
        gamma: float,
        gae_lambda: float,
        clip_grad: float | None,
        vf_coef: float,
        rew_norm: Callable[[Tensor], Tensor] | None,
        ent_coef: float,
        target_critic: polyak.Config | None,
        compute_dtype: torch.dtype,
    ):
        self.update_epochs = update_epochs
        self.mb_size = mb_size
        self.adv_norm = adv_norm
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_grad = clip_grad
        self.vf_coef = vf_coef
        self.rew_norm = rew_norm

        self.actor = actor
        self.critic = make_critic()

        if ac_forward is None:

            def ac_forward(actor, critics, obs):
                policy = actor(obs)
                values = tuple(critic(obs) for critic in critics)
                return policy, values

        self.ac_forward: ACForward = ac_forward

        parameters = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = make_opt(parameters)
        self.opt = ScaledOptimizer(self.opt, compute_dtype)

        self.ent_coef = ent_coef

        if target_critic is not None:
            self.critic_t = make_critic()
            self.critic_t.requires_grad_(False)
            polyak.sync(self.critic, self.critic_t)

            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.critic_t,
                tau=target_critic.tau,
                every=target_critic.every,
            )
        else:
            self.critic_t = None

        device = next(self.actor.parameters()).device
        self.autocast = lambda: torch.autocast(device.type, compute_dtype)

    @torch.no_grad()
    def preprocess(self, batch: list[RLSlices]) -> PPOData:
        lengths = torch.tensor([len(seq["obs"]) for seq in batch])
        end = torch.cumsum(lengths, 0)
        start = end - lengths

        obs = torch.cat([seq["obs"][:-1] for seq in batch])
        act = torch.cat([seq["act"] for seq in batch])

        all_obs = torch.cat([seq["obs"] for seq in batch])

        if self.critic_t is None:
            policy, (val,) = self.ac_forward(self.actor, (self.critic,), all_obs)
            val_t = val
        else:
            policy, (val, val_t) = self.ac_forward(
                self.actor, (self.critic, self.critic_t), all_obs
            )

        act_for_logp = []
        for seq in batch:
            act_for_logp.extend((seq["act"], seq["act"][:1]))
        act_for_logp = torch.cat(act_for_logp)
        logp = policy.log_prob(act_for_logp)

        logps, advs, rets, vals, weights = ([] for _ in range(5))
        for idx, seq in enumerate(batch):
            cont = 1.0 - seq["term"].float()
            seq_wt = torch.cat([torch.ones_like(cont[:1]), cont])
            seq_wt = torch.cumprod(seq_wt, 0)
            weights.append(seq_wt[:-2])

            gamma = self.gamma * cont
            seq_val_t = val_t[start[idx] : end[idx]]
            reward = seq["reward"]
            if self.rew_norm is not None:
                reward = self.rew_norm(reward)
            adv, ret = gen_adv_est(reward, seq_val_t, gamma, self.gae_lambda)
            advs.append(adv)
            rets.append(ret)

            logps.append(logp[start[idx] : end[idx] - 1])
            vals.append(val[start[idx] : end[idx] - 1])

        logp = torch.cat(logps)
        adv = torch.cat(advs)
        ret = torch.cat(rets)
        val = torch.cat(vals)
        weight = torch.cat(weights)

        return PPOData(obs, act, logp, adv, ret, val, weight)

    def opt_step(self, data: PPOData, return_metrics: bool = False):
        batch_size = len(data.val)

        for _ in range(self.update_epochs):
            if self.mb_size is None:
                splits = [slice(0, len(data.val))]
            else:
                num_mb = batch_size // self.mb_size
                splits = get_minibatches(batch_size, num_mb)
                perm = torch.randperm(batch_size)
                splits = [perm[idx] for idx in splits]

            for idxes in splits:
                weight = data.weight[idxes]

                with self.autocast():
                    new_policy, (new_val,) = self.ac_forward(
                        self.actor, (self.critic,), data.obs[idxes]
                    )
                    new_logp = new_policy.log_prob(data.act[idxes])
                    log_ratio = new_logp - data.logp[idxes]
                    ratio = log_ratio.exp()

                    adv_ = data.adv[idxes]
                    if self.adv_norm:
                        true_adv = adv_.clone()
                        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
                    else:
                        true_adv = adv_

                    t1 = -adv_ * ratio
                    t2 = -adv_ * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                    policy_loss = (weight * torch.max(t1, t2)).mean()

                    new_ent = new_policy.entropy()
                    ent_loss = self.ent_coef * (weight * -new_ent).mean()

                    actor_loss = policy_loss + ent_loss

                    if self.clip_vloss:
                        clipped_v = data.val[idxes] + (new_val - data.val[idxes]).clamp(
                            -self.clip_coef, self.clip_coef
                        )
                        v_losses1 = (new_val - data.ret[idxes]).square()
                        v_losses2 = (clipped_v - data.ret[idxes]).square()
                        v_losses = 0.5 * torch.max(v_losses1, v_losses2)
                    else:
                        v_losses = 0.5 * (new_val - data.ret[idxes]).square()
                    v_loss = self.vf_coef * (weight * v_losses).mean()

                    loss = actor_loss + v_loss

                self.opt.step(loss, clip_grad=self.clip_grad)

                if self.critic_t is not None:
                    self.update_target.step()

        if return_metrics:
            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                return {
                    "clip_frac": clip_frac,
                    "adv": true_adv.mean(),
                    "policy_loss": policy_loss.detach(),
                    "actor_loss": actor_loss.detach(),
                    "v_loss": v_loss.detach(),
                    "entropy": new_ent.mean(),
                    "value": data.val.mean(),
                }


class VecAgent(gym.vector.agents.Markov):
    def __init__(self, actor: Actor, mode: Literal["train", "val"]):
        super().__init__(actor.obs_space, actor.act_space)
        self.actor = actor
        self.mode = mode
        self._device = next(self.actor.parameters()).device

    def get_policy(self, last_obs: Tensor):
        last_obs = last_obs.to(self._device)
        policy: D.Distribution = self.actor(last_obs)
        if self.mode == "train":
            return policy.sample()
        elif self.mode == "val":
            return policy.mode


class Trainer:
    project = "ppo"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        self.setup_base()
        self.setup_envs()
        self.setup_ppo()

        should_sample = cron.Every(lambda: self.env_step, self.cfg.sample_every)

        while True:
            if should_sample:
                self.sample_episode()
            self.do_train_step()

    def setup_base(self):
        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)
        repro.seed_all(self.cfg.seed)

        self.exp = Experiment(
            project=self.project,
            create_commit=self.cfg.create_exp_commit,
        )
        self.exp.add_board(
            boards.Tensorboard(self.exp.dir / "board", launch=True),
        )

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.pbar = self.exp.make_pbar(desc=self.project)

    def setup_envs(self):
        self.sdk: sdk.atari.SDK = sdk.make(self.cfg.env)
        self.sdk = sdk.wrappers.ToTensor(self.sdk)

        if self.cfg.env.type == "atari":
            cfg = self.cfg.env.atari
            self._fps = 60 / cfg.frame_skip

        self.actor = Actor(self.sdk.obs_space, self.sdk.act_space)
        self.actor.to(self.device)

        self.train_envs = self.sdk.make_envs(self.cfg.num_train_envs, mode="train")
        self.train_agent = VecAgent(self.actor, mode="train")

        self.counters = defaultdict(lambda: 0)
        self.env_iter = iter(self.sdk.rollout(self.train_envs, self.train_agent))

        self.train_loader = OnPolicyRLLoader(
            do_env_step=self.do_env_step,
            temp_buf=self.sdk.wrap_buffer(Buffer()),
            steps_per_batch=self.cfg.steps_per_batch,
            min_seq_len=self.cfg.min_seq_len,
        )
        self.train_iter = iter(self.train_loader)

        num_val_envs = self.cfg.num_val_envs or self.cfg.num_train_envs
        self.val_envs = self.sdk.make_envs(num_val_envs, mode="val")
        self.val_agent = VecAgent(self.actor, mode="val")

        self.sample_val_env = self.sdk.make_envs(1, mode="val", render=True)
        self.sample_val_agent = VecAgent(self.actor, mode="val")
        self.sample_train_env = self.sdk.make_envs(1, mode="train", render=True)
        self.sample_train_agent = VecAgent(self.actor, mode="train")

    def do_env_step(self):
        env_idx, (step, final) = next(self.env_iter)

        delta = step["total_steps"] - self.counters[env_idx]
        self.env_step += delta
        self.counters[env_idx] = step["total_steps"]

        self.pbar.n = self.env_step
        self.pbar.refresh()

        if final:
            self.exp.add_scalar("train/ep_returns", step["ep_returns"])

        return env_idx, (step, final)

    def setup_ppo(self):
        if self.cfg.share_encoder:

            def make_critic():
                critic = CriticHead(self.actor.num_features)
                critic.to(self.device)
                return critic

            def ac_forward(actor: Actor, critics: tuple[CriticHead, ...], obs: Tensor):
                features = actor.encoder(obs)
                policy = actor.head(features)
                values = tuple(critic(features) for critic in critics)
                return policy, values

        else:

            def make_critic():
                critic = Critic(obs_space=self.sdk.obs_space)
                critic.to(self.device)
                return critic

            ac_forward = None

        make_opt = partial(torch.optim.Adam, lr=self.cfg.lr, eps=self.cfg.opt_eps)

        rew_norm_fns = {
            "id": None,
            "clip": lambda x: x.clamp(-1.0, 1.0),
            "sign": torch.sign,
        }

        self.ppo = PPO(
            actor=self.actor,
            make_critic=make_critic,
            ac_forward=ac_forward,
            make_opt=make_opt,
            update_epochs=self.cfg.update_epochs,
            mb_size=self.cfg.mb_size,
            adv_norm=self.cfg.adv_norm,
            clip_coef=self.cfg.clip_coef,
            clip_vloss=self.cfg.clip_vloss,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_grad=self.cfg.clip_grad,
            vf_coef=self.cfg.vf_coef,
            rew_norm=rew_norm_fns[self.cfg.rew_norm],
            ent_coef=self.cfg.ent_coef,
            target_critic=self.cfg.target_critic,
            compute_dtype=self.compute_dtype,
        )

    def sample_episode(self):
        modes = {
            "val": (self.sample_val_env, self.sample_val_agent),
            "train": (self.sample_train_env, self.sample_train_agent),
        }

        for mode in modes:
            env, agent = modes[mode]
            sample_iter = self.sdk.rollout(env, agent)

            frames = []
            for _, (step, final) in sample_iter:
                frame = np.asarray(step["render"].convert("RGB"))
                frames.append(frame)
                if final:
                    break

            clip = ImageSequenceClip(frames, fps=self._fps)
            self.exp.add_video(f"{mode}/episode", clip)

    def do_train_step(self):
        batch = next(self.train_iter)
        batch = [{k: v.to(self.device) for k, v in seq.items()} for seq in batch]

        data = self.ppo.preprocess(batch)
        metrics = self.ppo.opt_step(data, return_metrics=True)

        for k, v in metrics.items():
            self.exp.add_scalar(f"train/{k}", v)


def main():
    """Main function."""
    yaml = YAML(typ="safe", pure=True)
    with open(Path(__file__).parent / "config.yml", "r") as f:
        cfg = cast(yaml.load(f), Config)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
