from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from ..common import nets
from ..common.trainer import TrainerBase
from ..common.utils import find_class, null_ctx


@dataclass
class Config:
    actor: dict
    vf: dict

    opt: dict
    min_q: bool
    target_vf: dict | None
    coef: dict[str, float]
    clip_grad: float
    gamma: float
    gae_lambda: float
    use_adv_norm: bool
    actor_grad: Literal["dynamics", "reinforce", "mix"]
    actor_grad_mix: float | dict | None
    actor_ent: float | dict
    rew_norm: dict

    use_log_ent: bool
    """Whether to use log(H) for entropy loss. In the case of discrete action spaces, the entropy of degenerate policies vanishes, whereas in continuous action spaces, entropy goes to -inf. This (sometimes) makes entropy loss not aggressive enough in preventing the policy from collapsing."""


class Actor(nn.Sequential):
    def __init__(
        self,
        cfg: Config,
        state_size: int,
        act_space: spaces.torch.Tensor,
    ):
        mlp = nets.MLP(state_size, None, **cfg.actor)
        head = nets.ActorHead(mlp.out_features, act_space)
        super().__init__(mlp, head)


class ValueFunc(nets.BoxDecoder):
    def __init__(self, cfg: Config, state_size: int):
        vf_space = spaces.torch.Box((), dtype=torch.float32)
        super().__init__(state_size, vf_space, **cfg.vf)


def gen_adv_est(
    reward: Tensor,
    value: Tensor,
    cont: Tensor,
    gamma: float,
    gae_lambda: float,
):
    """Generalized advantage estimation.

    Given a sequence `s[0] -(a[0])-> s[1], r[0] -> ... -> s[T], r[T-1]`, with states `s[0:T+1]` having values `v[0:T+1]` and non-terminality indicators `c[0:T+1]`, computes GAE advantage and returns estimates adv[0:T], ret[0:T]. Returns estimate ret[t] corresponds to an improved value estimate for s[t]. Advantage estimate adv[t] corresponds to an advantage for performing action a[t] in state s[t].

    Args:
        reward: Reward sequence.
        value: Value sequence.
        cont: Non-terminality indicator sequence.
        gamma: Reward discount coefficient.
        gae_lambda: K-step discount coefficient.

    Returns:
        Tuple (adv, ret) of advantage and returns sequences.
    """

    # Compute advantage estimates using (16)
    delta = (reward + gamma * cont[1:] * value[1:]) - value[:-1]
    adv = [delta[-1]]
    for t in reversed(range(len(reward) - 1)):
        prev_adv = delta[t] + (gamma * gae_lambda) * cont[t + 1] * adv[-1]
        adv.append(prev_adv)
    adv.reverse()
    adv = torch.stack(adv)

    ret = value[:-1] + adv

    return adv, ret


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(
            clip_grad=cfg.clip_grad,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg

        self.rew_norm = nets.StreamNorm(**self.cfg.rew_norm)

        if self.cfg.actor_grad_mix is not None:
            self.actor_grad_mix = self._make_sched(self.cfg.actor_grad_mix)
        self.actor_ent_coef = self._make_sched(self.cfg.actor_ent)

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, dict):
            cfg = {**cfg}
            cls = getattr(sched, cfg["type"])
            del cfg["type"]
            return cls(**cfg)
        else:
            return sched.Constant(cfg)

    def setup(self, actor: Actor, make_vf: Callable[[], ValueFunc]):
        self.actor = actor

        if self.cfg.min_q:
            self.vf = nn.ModuleList([make_vf(), make_vf()])
        else:
            self.vf = make_vf()

        if self.cfg.target_vf is not None:
            if self.cfg.min_q:
                self.vf_t = nn.ModuleList([make_vf(), make_vf()])
            else:
                self.vf_t = make_vf()
            polyak.sync(self.vf, self.vf_t)
            self.update_vf_t = polyak.Polyak(self.vf, self.vf_t, **self.cfg.target_vf)

        self.parameters = [*self.actor.parameters(), *self.vf.parameters()]

        self.opt = self._make_opt()

        self.opt_iter = 0

    def _make_opt(self):
        cfg = {**self.cfg.opt}

        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]

        actor, vf = cfg["actor"], cfg["vf"]
        del cfg["actor"]
        del cfg["vf"]

        common = cfg

        return cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.vf.parameters(), **vf},
            ],
            **common,
        )

    def compute(self, states, action, reward, term):
        losses, mets = {}, {}

        with torch.no_grad():
            with self.autocast():
                if self.cfg.min_q:
                    v_t = torch.minimum(
                        over_seq(self.vf_t[0])(states).mode,
                        over_seq(self.vf_t[1])(states).mode,
                    )
                else:
                    v_t = over_seq(self.vf_t)(states).mode

                cont = 1.0 - term
                reward = self.rew_norm(reward)

                is_real = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])
                is_real = is_real.cumprod(0)

        # If actor_grad is set to 'reinforce', we must compute gen_adv_est in torch.no_grad() - otherwise, autograd will leak memory due to gradients not being freed, since adv will be detached.
        if self.cfg.actor_grad == "reinforce":
            gen_adv_ctx = torch.no_grad()
        else:
            gen_adv_ctx = null_ctx()

        with gen_adv_ctx:
            with self.autocast():
                adv, returns = gen_adv_est(
                    reward=reward,
                    value=v_t,
                    cont=cont,
                    gamma=self.cfg.gamma,
                    gae_lambda=self.cfg.gae_lambda,
                )

        with self.autocast():
            if self.cfg.use_adv_norm:
                adv = (adv - adv.mean()) / adv.std().clamp_min(1e-8)

            states_sg = states[:-1].detach()
            policy = over_seq(self.actor)(states_sg)

            if self.cfg.actor_grad == "dynamics":
                objective = returns
            elif self.cfg.actor_grad == "reinforce":
                objective = adv.detach() * policy.log_prob(action.detach())
            elif self.cfg.actor_grad == "mix":
                dyn_obj = returns
                vpg_obj = adv.detach() * policy.log_prob(action.detach())
                mix = self.actor_grad_mix(self.opt_iter)
                objective = mix * dyn_obj + (1.0 - mix) * vpg_obj

            ent_coef = self.actor_ent_coef(self.opt_iter)
            mets["ent_coef"] = ent_coef
            ent_value = policy.entropy()
            mets["entropy"] = ent_value.detach().mean()
            if self.cfg.use_log_ent:
                ent_value = ent_value.log()
            actor_loss = objective + ent_coef * ent_value
            losses["actor"] = -(is_real[:-1] * actor_loss).mean()

            vf_target = returns.detach()
            if self.cfg.min_q:
                vf1_dist = over_seq(self.vf[0])(states_sg)
                mets["value"] = (vf1_dist.detach().mean).mean()
                vf1_loss = -vf1_dist.log_prob(vf_target)
                vf2_dist = over_seq(self.vf[1])(states_sg)
                vf2_loss = -vf2_dist.log_prob(vf_target)
                vf_loss = vf1_loss + vf2_loss
            else:
                vf_dist = over_seq(self.vf)(states_sg)
                mets["value"] = (vf_dist.detach().mean).mean()
                vf_loss = -vf_dist.log_prob(vf_target)
            losses["vf"] = (is_real[:-1] * vf_loss).mean()

            for k, v in losses.items():
                mets[f"{k}_loss"] = v.detach()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        return loss, mets

    def opt_step(self, loss: Tensor):
        super().opt_step(loss)
        if self.cfg.target_vf is not None:
            self.update_vf_t.step()
        self.opt_iter += 1
