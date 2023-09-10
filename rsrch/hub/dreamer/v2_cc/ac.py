import torch
from torch import nn

from rsrch.rl import gym
from rsrch.rl.utils.polyak import Polyak
from rsrch.types import Namespace

from .nets import FC


class StreamNorm:
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        self._shape = shape
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.reset()

    def reset(self):
        self.mag = torch.ones(self._shape)

    def update(self, inputs):
        cur = inputs.abs().reshape(-1, *self._shape).mean(0)
        self.mag = self._momentum * self.mag + (1.0 - self._momentum) * cur

    def transform(self, x):
        x = x.reshape(-1, *self._shape)
        x = self._scale * (x / self.mag.type_as(x) + self._eps)
        return x


def lambda_return(rew, v, disc, bootstrap, lambda_):
    disc = torch.as_tensor(disc, dtype=rew.dtype, device=rew.device)
    disc = disc.broadcast_to(rew.shape)
    if bootstrap is None:
        bootstrap = torch.zeros_like(v[-1])
    next_values = torch.cat([v[1:], bootstrap], 0)
    inputs = rew + disc * next_values * (1.0 - lambda_)
    seq_len = len(rew)
    returns, cur_v = [], bootstrap
    for step in reversed(range(seq_len)):
        next_v = inputs[step] + disc[step] * lambda_ * cur_v
        returns.append(next_v)
        cur_v = next_v
    returns.reverse()
    return torch.stack(returns)


class Schedule:
    def __init__(self, name: str, **params):
        self._name = name
        self._params = Namespace(params)

    def forward(self, step):
        p = self._params
        if self._name == "const":
            return p.value
        elif self._name == "linear":
            mix = min(max(step / p.duration, 0.0), 1.0)
            mix = p.initial * (1.0 - mix) + p.final * mix
            return mix
        elif self._name == "warmup":
            mix = min(max(step / p.warmup, 0.0), 1.0)
            return mix * p.value
        elif self._name == "exp":
            amp = p.final - p.initial
            return amp * 0.5 ** (step / p.halflife) + p.final
        elif self._name == "horizon":
            mix = min(max(step / p.duration, 0.0), 1.0)
            hor = (1.0 - mix) * p.initial + mix * p.final
            return 1.0 - 1.0 / hor
        else:
            raise NotImplementedError(self._name)


class Optimizer(nn.Module, torch.optim.Optimizer):
    def __init__(self, nets: list, name: str, **args):
        super().__init__()
        self._nets = nets
        self._name = name
        self._args = args
        self.init()

    def init(self):
        params = [p for net in self._nets for p in net.parameters()]
        if self._name == "adam":
            self._opt = torch.optim.Adam(params, **self._args)
        else:
            raise NotImplementedError(self._name)

    def zero_grad(self, *args, **kwargs):
        self._opt.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._opt.step(*args, **kwargs)

    def to(self, device=None):
        # If we move the parent class, we should reconstruct the optimizer
        self.init()


class ActorCritic(nn.Module):
    def __init__(self, config, act_space, state_dim, step_fn):
        super().__init__()
        self.config = config
        self.step_fn = step_fn

        assert isinstance(act_space, gym.spaces.TensorSpace)
        discrete = isinstance(act_space, gym.spaces.TensorDiscrete)
        if self.config.actor.dist.name == "auto":
            self.config.actor.dist.name = "onehot" if discrete else "trunc_normal"
        if self.config.actor_grad == "auto":
            self.config.actor_grad = "reinforce" if discrete else "dynamics"

        self.actor = FC(state_dim, act_space.shape, **self.config.actor)
        self.actor_opt = Optimizer([self.actor], **self.config.actor_opt)
        self.critic = FC(state_dim, [], **self.config.critic)
        self.critic_opt = Optimizer([self.critic], **self.config.critic_opt)

        if self.config.slow_target.enabled:
            self._target_critic = FC(state_dim, [], **self.config.critic)
            self.critic_polyak = Polyak(
                self.critic,
                self._target_critic,
                tau=self.config.slow_target.tau,
                every=self.config.slow_target.every,
            )
        else:
            self._target_critic = self.critic

        self.rew_norm = StreamNorm(**self.config.rew_norm)
        self.mix_sched = Schedule(**self.config.actor_grad_mix)
        self.ent_sched = Schedule(**self.config.actor_ent)

    def train(self, world_model, start, is_terminal, reward_fn):
        hor = self.config.imag_horizon
        seq = world_model.imagine(self.actor, start, is_terminal, hor)
        seq.reward = self.rew_norm(reward_fn(seq))
        vt = self.target(seq)

        actor_loss = self.actor_loss(seq, vt)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        critic_loss = self.critic_loss(seq, vt)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

    def actor_loss(self, seq, vt):
        policy = self.actor(seq.feat[:-2].detach())
        if self.config.actor_grad == "dynamics":
            objective = vt[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq.feat[:-2]).mode()
            adv = (vt[1:] - baseline).detach()
            action = seq.action[1:-1].detach()
            objective = policy.log_prob(action) * adv
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq.feat[:-2]).mode()
            adv = (vt[1:] - baseline).detach()
            action = seq.action[1:-1].detach()
            objective = policy.log_prob(action) * adv
            mix = self.mix_sched(self.step_fn())
            objective = mix * vt[1:] + (1.0 - mix) * objective
        else:
            raise NotImplementedError(self.config.actor_grad)

        ent = policy.entropy()
        ent_scale = self.ent_sched(self.step_fn())
        objective = objective + ent_scale * ent

        actor_loss = -(seq.weight[:-2].detach() * objective).mean()
        return actor_loss

    def critic_loss(self, seq, vt):
        dist = self.critic(seq.feat[:-1])
        w = seq.weight[:-1].detach()
        critic_loss = -(dist.log_prob(vt.detach()) * w).mean()
        return critic_loss

    def target(self, seq):
        value = self._target_critic(seq.feat).mode()
        target = lambda_return(
            rew=seq.reward[:-1],
            v=value[:-1],
            disc=seq.discount[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
        )
        return target
