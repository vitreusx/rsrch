from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from config import Config
from pydantic import BaseModel
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from rsrch import rl
from rsrch.exp import boards
from rsrch.exp.experiment import Experiment
from rsrch.jax.distributions import Categorical
from rsrch.jax.utils import key_seq
from rsrch.rl import data, sdk
from rsrch.rl.loaders import OnPolicyRLLoader
from rsrch.rl.sdk.wrappers.jax import ToArray

if TYPE_CHECKING:
    from rsrch.rl.utils import polyak
    from rsrch.spaces import jax as spaces_jax


class Apply(eqx.Module):
    fn: Any

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x: jax.Array, key=None):  # noqa: ARG002
        return self.fn(x)


class AtariEncoder(eqx.nn.Sequential):
    out_features: int = 512

    def __init__(self, space: spaces_jax.Image, *, key: jax.Array):
        if space.size != (84, 84):
            msg = "AtariEncoder only supports images of size (84, 84)"
            raise ValueError(msg)

        keys = iter(key_seq(key))
        super().__init__(
            [
                eqx.nn.Conv2d(space.num_channels, 32, 8, 4, key=next(keys)),
                Apply(jax.nn.relu),
                eqx.nn.Conv2d(32, 64, 4, 2, key=next(keys)),
                Apply(jax.nn.relu),
                eqx.nn.Conv2d(64, 64, 3, 1, key=next(keys)),
                Apply(jax.nn.relu),
                Apply(jnp.ravel),
                eqx.nn.Linear(64 * 7 * 7, self.out_features, key=next(keys)),
                Apply(jax.nn.relu),
            ]
        )


class Actor(eqx.Module):
    encoder: AtariEncoder
    head: eqx.nn.Linear

    def __init__(
        self,
        obs_space: spaces_jax.Image,
        act_space: spaces_jax.Discrete,
        *,
        key: jax.Array,
    ):
        super().__init__()
        keys = iter(key_seq(key))
        self.encoder = AtariEncoder(obs_space, key=next(keys))
        self.head = eqx.nn.Linear(
            self.encoder.out_features, act_space.n, key=next(keys)
        )

    def __call__(self, obs: jax.Array, state: eqx.nn.State, *, key=None):
        features, state = self.encoder(obs, state, key=key)
        logits = self.head(features)
        dist = Categorical(logits=logits)
        return dist, state


class Critic(eqx.Module):
    encoder: AtariEncoder
    head: eqx.nn.Linear

    def __init__(
        self,
        obs_space: spaces_jax.Image,
        *,
        key: jax.Array,
    ):
        super().__init__()
        keys = iter(key_seq(key))
        self.encoder = AtariEncoder(obs_space, key=next(keys))
        self.head = eqx.nn.Linear(self.encoder.out_features, 1, key=next(keys))

    def __call__(self, obs: jax.Array, state: eqx.nn.State, *, key=None):
        features, state = self.encoder(obs, state, key=key)
        value = self.head(features).reshape(())
        return value, state


class Agent(rl.VecAgent):
    def __init__(
        self,
        actor: Actor,
        state: eqx.nn.State,
        obs_space: spaces_jax.Array,
        act_space: spaces_jax.Array,
        mode: Literal["train", "val"] = "train",
        key: jax.Array | None = None,
    ):
        super().__init__(obs_space, act_space)
        self.batch_actor = jax.jit(
            jax.vmap(actor, in_axes=(0, None), out_axes=(0, None))
        )
        self.state = state
        if key is not None:
            self.keys = iter(key_seq(key))
        else:
            self.keys = None
        self.mode = mode
        self._obs = None

    def reset(self, idxes, obs_seq: jax.Array):
        if self._obs is None:
            self._obs = obs_seq
        else:
            self._obs = self._obs.at[idxes].set(obs_seq, unique_indices=True)

    def policy(self, idxes):
        dist, _ = self.batch_actor(self._obs[idxes], self.state)
        if self.mode == "train":
            return dist.sample(key=next(self.keys))
        else:
            return dist.mode()

    def step(self, idxes, act_seq, next_obs_seq):
        return super().step(idxes, act_seq, next_obs_seq)


def get_minibatches(batch_size: int, num_mb: int):
    """Divide a batch into a number of minibatches.

    The minibatch sizes are selected in such a way, that they are divisible by 32,
    except for the last one. The last batch may be larger than the previous ones.
    """

    warp_size = 32
    batch_size_w = batch_size // warp_size
    mb_size = warp_size * (batch_size_w // num_mb)
    mb_size_rem = batch_size - num_mb * mb_size
    split_sizes = [mb_size] * num_mb
    split_sizes[-1] += mb_size_rem
    end = np.cumsum(split_sizes)
    start = end - np.array(split_sizes)
    return [slice(start_i, end_i) for start_i, end_i in zip(start, end, strict=True)]


class PPOConfig(BaseModel):
    pass


class ACForward(Protocol):
    def __call__(
        self,
        actor: Actor,
        critics: tuple[Critic, ...],
        obs: jax.Array,
        state: eqx.nn.State,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Any, tuple[jax.Array, ...], eqx.nn.State]:
        pass


class MakeCritic(Protocol):
    def __call__(self, *, key: jax.Array) -> Critic:
        pass


class RLSlices(eqx.Module):
    obs: jax.Array
    act: jax.Array
    reward: jax.Array
    term: jax.Array


class PPOData(eqx.Module):
    obs: jax.Array
    act: jax.Array
    logp: jax.Array
    adv: jax.Array
    ret: jax.Array
    val: jax.Array
    weight: jax.Array | None


@jax.jit
def gen_adv_est(
    reward: jax.Array,
    value: jax.Array,
    gamma: jax.Array,
    gae_lambda: float,
):
    r"""Generalized Advantage Estimation (GAE).

    :param reward: Array :math:`r_{1:L}` of shape :math:`(L)` of rewards obtained
        upon reaching a state.
    :param value: Array :math:`v_{0:L} of shape :math:`(L+1)` of value
        estimates.
    :param gamma: Array :math:`\gamma_{0:L}` of shape :math:`(L+1)` of the discount
        values for each state. Usually, :math:`\gamma_t = \gamma` for
        non-terminal states, and :math:`\gamma_t = 0` for terminal and
        post-terminal states.
    :param gae_lambda: :math:`\lambda` discount value.
    """

    delta = (reward + gamma[1:] * value[1:]) - value[:-1]

    def scan_fn(prev_adv, cur):
        gamma, delta = cur
        next_adv = delta + gae_lambda * gamma * prev_adv
        return next_adv, next_adv

    _, adv = jax.lax.scan(
        scan_fn,
        init=delta[-1],
        xs=(gamma[1:-1], delta[:-1]),
        reverse=True,
    )
    adv = jnp.concat((delta[-1:], adv))
    ret = value[:-1] + adv

    return adv, ret


class PPOOutput(eqx.Module):
    actor: Actor
    critic: Critic
    opt_state: Any
    state: eqx.nn.State
    loss: jax.Array
    true_adv: jax.Array


class PPO:
    def __init__(
        self,
        *,
        actor: Actor,
        make_critic: MakeCritic,
        opt: optax.GradientTransformation,
        key: jax.Array,
        custom_fwd: ACForward | None = None,
        update_epochs: int,
        mb_size: int | None,
        adv_norm: bool,
        clip_coef: float,
        clip_vloss: bool,
        gamma: float,
        gae_lambda: float,
        clip_grad: float | None,
        vf_coef: float,
        transform_reward: Callable[[jax.Array], jax.Array] | None = None,
        ent_coef: float,
        target_critic: polyak.Config | None = None,
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
        self.transform_reward = transform_reward
        self.ent_coef = ent_coef

        self.actor = actor
        self.critic = make_critic(key=key)
        if custom_fwd is not None:
            self.ac_forward = custom_fwd
        else:
            self.ac_forward = self.default_ac_forward

        self.batch_ac_forward = eqx.filter_jit(
            jax.vmap(
                self.ac_forward,
                in_axes=(None, None, 0, None, None),
                out_axes=(0, 0, None),
            )
        )

        self.opt = opt
        models = (self.actor, self.critic)
        params = eqx.filter(models, eqx.is_inexact_array)
        self.opt_state = self.opt.init(params)

        self.target_critic = target_critic
        if self.target_critic is not None:
            self.critic_t = make_critic(key=key)
        else:
            self.critic_t = None

    @staticmethod
    def default_ac_forward(
        actor: Actor,
        critics: tuple[Critic, ...],
        obs: jax.Array,
        state: eqx.nn.State,
        key: jax.Array | None = None,
    ):
        keys = iter(key_seq(key))
        policy, state = actor(obs, state, key=next(keys))
        values = []
        for critic in critics:
            value, state = critic(obs, state, key=next(keys))
            values.append(value)
        return policy, tuple(values), state

    def preprocess(
        self,
        batch: list[RLSlices],
        state: eqx.nn.State,
        *,
        key: jax.Array | None = None,
    ):
        obs = jnp.concat([seq.obs[:-1] for seq in batch])
        all_obs = jnp.concat([seq.obs for seq in batch])
        act = jnp.concat([seq.act for seq in batch])

        if self.critic_t is None:
            policy, (val,), _ = self.batch_ac_forward(
                self.actor, (self.critic,), all_obs, state, key
            )
            val_t = val
        else:
            policy, (val, val_t), _ = self.batch_ac_forward(
                self.actor, (self.critic, self.critic_t), all_obs, state, key
            )

        act_for_logp = []
        for seq in batch:
            act_for_logp.extend((seq.act, seq.act[:1]))
        act_for_logp = jnp.concat(act_for_logp)
        logp = policy.log_prob(act_for_logp)

        logps, advs, rets, vals, weights = ([] for _ in range(5))
        offset = 0

        for seq in batch:
            cont = 1.0 - seq.term.astype(jnp.float32)
            seq_wt = jnp.concat([jnp.ones_like(cont[:1]), cont])
            seq_wt = jnp.cumprod(seq_wt, 0)
            weights.append(seq_wt[:-2])

            seq_len = len(seq.obs)
            seq_val_t = val_t[offset : offset + seq_len]
            seq_logp = logp[offset : offset + seq_len - 1]
            seq_val = val[offset : offset + seq_len - 1]
            offset += seq_len

            logps.append(seq_logp)
            vals.append(seq_val)

            reward = seq.reward
            if self.transform_reward is not None:
                reward = self.transform_reward(reward)
            gamma = self.gamma * cont
            adv, ret = gen_adv_est(reward, seq_val_t, gamma, self.gae_lambda)
            advs.append(adv)
            rets.append(ret)

        logp = jnp.concat(logps)
        adv = jnp.concat(advs)
        ret = jnp.concat(rets)
        val = jnp.concat(vals)
        weight = jnp.concat(weights)

        return PPOData(obs, act, logp, adv, ret, val, weight)

    @eqx.filter_jit
    def opt_step(
        self,
        actor: Actor,
        critic: Critic,
        opt: optax.GradientTransformation,
        opt_state: Any,
        data: PPOData,
        state: eqx.nn.State,
        *,
        key: jax.Array | None = None,
    ) -> PPOOutput:
        batch_size = len(data.val)
        keys = key_seq(key)

        @eqx.filter_value_and_grad(has_aux=True)
        def compute_loss(
            models: tuple[Actor, Critic],
            aux: tuple[PPOData, eqx.nn.State, jax.Array],
        ):
            actor, critic = models
            data, state, key = aux

            new_policy, (new_val,), state = self.batch_ac_forward(
                actor, (critic,), data.obs, state, key=key
            )
            new_logp = new_policy.log_prob(data.act)
            log_ratio = new_logp - data.logp
            ratio = jnp.exp(log_ratio)

            adv_ = data.adv
            if self.adv_norm:
                true_adv = adv_
                adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
            else:
                true_adv = adv_

            t1 = -adv_ * ratio
            t2 = -adv_ * ratio.clip(1.0 - self.clip_coef, 1.0 + self.clip_coef)
            policy_losses = jnp.maximum(t1, t2)

            new_ent = new_policy.entropy()
            ent_losses = -new_ent

            if self.clip_vloss:
                clipped_v = data.val + (new_val - data.val).clip(
                    -self.clip_coef, self.clip_coef
                )
                v_losses1 = jnp.square(new_val - data.ret)
                v_losses2 = jnp.square(clipped_v - data.ret)
                v_losses = 0.5 * jnp.maximum(v_losses1, v_losses2)
            else:
                v_losses = 0.5 * jnp.square(new_val - data.ret)

            if data.weight is None:
                actor_loss = policy_losses.mean() + self.ent_coef * ent_losses.mean()
                critic_loss = v_losses.mean()
            else:
                actor_losses = policy_losses + self.ent_coef * ent_losses
                actor_loss = (data.weight * actor_losses).mean()
                critic_loss = (data.weight * v_losses).mean()

            loss = actor_loss + self.vf_coef * critic_loss
            return loss, (state, true_adv)

        for _ in range(self.update_epochs):
            if self.mb_size is None:
                splits = [slice(0, len(data.val))]
            else:
                num_mb = batch_size // self.mb_size
                splits = get_minibatches(batch_size, num_mb)
                perm = jax.random.permutation(next(keys), num_mb)
                splits = [perm[idx] for idx in splits]

            for idxes in splits:
                minibatch = PPOData(
                    obs=data.obs[idxes],
                    act=data.act[idxes],
                    logp=data.logp[idxes],
                    adv=data.adv[idxes],
                    ret=data.ret[idxes],
                    val=data.val[idxes],
                    weight=None if data.weight is None else data.weight[idxes],
                )

                (loss, (state, true_adv)), grads = compute_loss(
                    (actor, critic),
                    (minibatch, state, next(keys)),
                )

                updates, opt_state = opt.update(grads, opt_state, (actor, critic))
                actor, critic = eqx.apply_updates((actor, critic), updates)

        return PPOOutput(
            actor=actor,
            critic=critic,
            opt_state=opt_state,
            state=state,
            loss=loss,
            true_adv=true_adv,
        )


def jax_stack(items: list):
    if isinstance(items[0], jax.Array):
        return jnp.stack(items)
    else:
        return jnp.asarray(items)


class Trainer:
    project = "ppo-jax"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        self.setup_base()
        self.setup_envs()
        self.setup_agent()
        self.setup_ppo()

        while True:
            self.do_train_step()

    def setup_base(self):
        self.keys = iter(key_seq(jax.random.key(self.cfg.seed)))

        self.exp = Experiment(project=self.project)
        self.exp.add_board(boards.MLflow(exp_name=self.exp.project))

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)

    def setup_envs(self):
        self.sdk = sdk.make(self.cfg.env)
        self.sdk = ToArray(self.sdk)

        self.obs_space = self.sdk.obs_space
        self.act_space = self.sdk.act_space

        self.train_envs = self.sdk.make_envs(self.cfg.num_envs, mode="train")
        self.val_envs = self.sdk.make_envs(self.cfg.num_envs, mode="val")

    def setup_agent(self):
        self.actor, self.state = eqx.nn.make_with_state(Actor)(
            obs_space=self.obs_space,
            act_space=self.act_space,
            key=next(self.keys),
        )

        self.train_agent = Agent(
            actor=self.actor,
            state=self.state,
            obs_space=self.sdk.obs_space,
            act_space=self.sdk.act_space,
            mode="train",
            key=next(self.keys),
        )

        self.env_iter = iter(self.sdk.rollout(self.train_envs, self.train_agent))
        self.counters = defaultdict(lambda: 0)
        self.pbar = tqdm(desc="Env steps")

        self.train_loader = OnPolicyRLLoader(
            do_env_step=self.do_env_step,
            temp_buf=self.sdk.wrap_buffer(data.Buffer()),
            steps_per_batch=self.cfg.steps_per_batch,
            min_seq_len=self.cfg.min_seq_len,
            stack_fn=jax_stack,
        )
        self.train_iter = iter(self.train_loader)

        self.val_agent = Agent(
            actor=self.actor,
            state=self.state,
            obs_space=self.sdk.obs_space,
            act_space=self.sdk.act_space,
            mode="val",
            key=None,
        )

    def do_env_step(self):
        self.train_agent.state = self.state
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
        def make_critic(key: jax.Array):
            return Critic(self.sdk.obs_space, key=key)

        rew_norm_fns = {
            "id": None,
            "clip": lambda x: x.clip(-1.0, 1.0),
            "sign": jnp.sign,
        }

        self.opt = optax.adamw(
            learning_rate=self.cfg.adamw_lr,
            eps=self.cfg.adamw_eps,
        )

        self.ppo = PPO(
            actor=self.actor,
            make_critic=make_critic,
            opt=self.opt,
            key=next(self.keys),
            update_epochs=self.cfg.update_epochs,
            mb_size=self.cfg.mb_size,
            adv_norm=self.cfg.adv_norm,
            clip_coef=self.cfg.clip_coef,
            clip_vloss=self.cfg.clip_vloss,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_grad=self.cfg.clip_grad,
            vf_coef=self.cfg.vf_coef,
            ent_coef=self.cfg.ent_coef,
            transform_reward=rew_norm_fns[self.cfg.rew_transform],
            target_critic=self.cfg.target_critic,
        )

        params = eqx.filter((self.actor, self.ppo.critic), eqx.is_inexact_array)
        self.opt_state = self.opt.init(params)

    def do_train_step(self):
        batch = next(self.train_iter)
        batch = [RLSlices(**seq) for seq in batch]
        data = self.ppo.preprocess(batch, self.state, key=next(self.keys))
        output = self.ppo.opt_step(
            actor=self.actor,
            critic=self.ppo.critic,
            opt=self.opt,
            opt_state=self.opt_state,
            data=data,
            state=self.state,
            key=next(self.keys),
        )

        self.actor, self.ppo.critic = output.actor, output.critic
        self.opt_state = output.opt_state
        self.state = output.state

        self.exp.add_scalar("train/loss", output.loss)


class Args(BaseModel):
    config: str | None


def main():
    args = tyro.cli(Args)

    yaml = YAML(typ="safe", pure=True)
    with open(args.config) as f:
        cfg_d = yaml.load(f)

    cfg = Config(**cfg_d)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
