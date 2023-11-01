from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch.exp.profiler import Profiler
from rsrch.exp.wandb import Experiment
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron

from . import config, env
from .config import Config
from .wm.rnn import WorldModel


@cache
def over_seq(_func):
    """Transform a function that operates on batches (B, ...) to operate on
    sequences (L, B, ...)."""

    def _lifted(x: Tensor, *args, **kwargs):
        batch_size = x.shape[1]
        y = _func(x.flatten(0, 1), *args, **kwargs)
        return y.reshape(-1, batch_size, *y.shape[1:])

    return _lifted


class WMAgent(gym.vector.Agent):
    def __init__(self, env_f, wm: WorldModel, actor):
        super().__init__()
        self.env_f = env_f
        self.wm = wm
        self.actor = actor
        self._state = None

    def reset(self, idxes, obs, info):
        obs = self.env_f.load_obs(obs)
        if self._state is None:
            self._state = self.wm.init(obs)
        else:
            self._state[:, idxes] = self.wm.init(obs)

    def policy(self, obs):
        act = self.actor(self._state)
        return act.cpu().numpy()

    def step(self, act):
        act = self.env_f.load_act(act)
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.env_f.load_obs(next_obs)
        trans_x = torch.cat([self._act[idxes], next_obs], -1)
        trans_h0 = self._state[:, idxes]
        _, next_s = self.wm.trans(trans_x[None], trans_h0)
        self._state[:, idxes] = next_s


class EpsRandomAgent(gym.vector.Agent):
    def __init__(self, opt, rand, eps):
        super().__init__()
        self.opt = opt
        self.rand = rand
        self.eps = eps

    def reset(self, idxes, obs, info):
        for agent in [self.opt, self.rand]:
            agent.reset(idxes, obs, info)

    def policy(self, obs):
        if torch.rand(1) < self.eps:
            return self.rand.policy(obs)
        else:
            return self.opt.policy(obs)

    def step(self, act):
        for agent in [self.opt, self.rand]:
            agent.step(act)

    def observe(self, idxes, next_obs, term, trunc, info):
        for agent in [self.opt, self.rand]:
            agent.observe(idxes, next_obs, term, trunc, info)


class CEMPlanner:
    def __init__(
        self,
        cfg: Config.CEM,
        wm: WorldModel,
        act_space: gym.TensorSpace,
    ):
        self.cfg = cfg
        self.wm = wm
        self.act_space = act_space

    @torch.inference_mode()
    def __call__(self, h: Tensor):
        num_envs = h.shape[1]
        if isinstance(self.act_space, gym.spaces.TensorBox):
            act_shape = self.act_space.shape
            seq_shape = [num_envs, self.cfg.horizon, *act_shape]  # [E, H, *A]
            seq_loc = torch.zeros(seq_shape, dtype=h.dtype, device=h.device)
            seq_scale = torch.ones(seq_shape, dtype=h.dtype, device=h.device)
            seq_rv = D.Normal(seq_loc, seq_scale, len(seq_shape))
        else:
            act_shape = []
            seq_probs = torch.ones(
                [num_envs, self.cfg.horizon, self.act_space.n],
                dtype=h.dtype,
                device=h.device,
            )  # [E, H, #A]
            seq_probs = seq_probs / self.act_space.n
            seq_rv = D.Categorical(probs=seq_probs)

        # h.shape = [L, E, D_h]
        pred_h0 = h.repeat(1, self.cfg.pop, 1)  # [L, E*P, D_h]

        for _ in range(self.cfg.niters):
            act = seq_rv.sample([self.cfg.pop])  # [P, E, H, *A]
            act = act.flatten(0, 1).swapaxes(0, 1)  # [H, E*P, *A]
            enc_act = over_seq(self.wm.act_enc)(act)  # [H, E*P, D_a]

            pred_x = enc_act
            pred_hx, _ = self.wm.pred(pred_x, pred_h0)  # [H, E*P, D_h]
            rew_rv = over_seq(self.wm.rew)(pred_hx)  # [H, E*P]
            term_rv = over_seq(self.wm.term)(pred_hx)
            term: Tensor = term_rv.mean  # [H, E*P]
            cont = (1 - term).cumprod(0)  # [H, E*P]
            ret = (rew_rv.mean * cont).sum(0)  # [E*P]
            ret = ret.reshape(self.cfg.pop, num_envs)  # [P, E]

            elite_idx = torch.topk(ret, k=self.cfg.elites, dim=0).indices  # [El, E]
            act_ = act.reshape(self.cfg.horizon, self.cfg.pop, num_envs, *act_shape)
            elite_idx = elite_idx.reshape(1, *elite_idx.shape, *act_shape)
            elite_idx = elite_idx.expand(self.cfg.horizon, *elite_idx.shape[1:])
            # act_: [H, P, E, *A], elite_idx: [H, El, E, *A]
            elites = act_.gather(1, elite_idx)  # [H, El, E, *A]

            if isinstance(self.act_space, gym.spaces.TensorBox):
                elite_loc = elites.mean(1).swapaxes(0, 1)  # [E, H, *A]
                elite_scale = elites.std(1).swapaxes(0, 1)  # [E, H, *A]
                seq_rv = D.Normal(elite_loc, elite_scale, len(seq_shape))
            else:
                elites01 = nn.functional.one_hot(elites, self.act_space.n)
                elite_probs = elites01.float().mean(1).swapaxes(0, 1)  # [E, H, #A]
                seq_rv = D.Categorical(probs=elite_probs)

        return seq_rv.mode[0]  # [*A]


def main():
    cfg_dict = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(
        data=cfg_dict,
        cls=config.Config,
    )

    device = torch.device(cfg.device)

    env_f = env.make_factory(cfg.env, device)
    sampler = data.UniformSampler()
    buffer = env_f.chunk_buffer(cfg.capacity, cfg.seq_len, sampler)

    obs_dim = env_f.obs_space.shape[0]
    act_dim = env_f.act_space.n
    wm = WorldModel(obs_dim, act_dim, 64, 4)
    wm = wm.to(device)

    wm_opt = cfg.wm_opt.make()(wm.parameters())

    envs = env_f.vector_env(1, mode="train")
    actor = CEMPlanner(cfg.cem, wm, env_f.act_space)
    agent = EpsRandomAgent(
        opt=WMAgent(env_f, wm, actor),
        rand=gym.vector.agents.RandomAgent(envs),
        eps=0.0,
    )
    env_iter = iter(rollout.steps(envs, agent))
    ep_ids = [None for _ in range(envs.num_envs)]
    ep_rets = [0.0 for _ in range(envs.num_envs)]

    env_step = 0
    exp = Experiment(project="pets_v2", config=cfg_dict)
    board = exp.board
    board.add_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.total_steps)

    should_log = cron.Every(lambda: env_step, 128)
    prof = Profiler(
        cfg=cfg.profiler,
        device=device,
        step_fn=lambda: env_step,
        trace_path=exp.dir / "trace.json",
    )

    def update_env_step(n=1):
        nonlocal env_step
        env_step += n
        pbar.update(n)
        prof.update()
        # agent.eps = 1.0 if len(buffer) < cfg.prefill else 0.0
        # agent.eps = np.clip(1.0 - 4.0 * env_step / cfg.total_steps, 0.0, 1.0)

    while env_step < cfg.total_steps:
        env_idx, step = next(env_iter)
        ep_ids[env_idx], _ = buffer.push(ep_ids[env_idx], step)
        ep_rets[env_idx] += step.reward
        if step.done:
            board.add_scalar("train/ep_ret", ep_rets[env_idx])
            ep_rets[env_idx] = 0.0

        update_env_step()

        if len(buffer) < cfg.prefill:
            continue

        for _ in range(4):
            idxes = sampler.sample(cfg.batch_size)
            batch = env_f.fetch_chunk_batch(buffer, idxes)

            obs = batch.obs
            act = over_seq(wm.act_enc)(batch.act)
            L, N = act.shape[:2]

            trans_h0 = wm.init(obs[0])  # [#Layer, N, H]
            trans_x = torch.cat([act, obs[1:]], -1)  # [L, N, D_act + D_obs]
            trans_hx, _ = wm.trans(trans_x, trans_h0)  # [L, N, H]
            trans_hx = torch.cat([trans_h0[[-1]], trans_hx], 0)  # [L + 1, N, H]

            idxes = torch.randint(0, L, (N,))
            pred_h0 = trans_hx[idxes, np.arange(N)]
            pred_x = act[idxes, np.arange(N)].unsqueeze(0)
            pred_h1 = trans_hx[idxes + 1, np.arange(N)]
            # pred_h0 = trans_hx[:-1].flatten(0, 1)  # [L*N, H]
            # pred_x = act.reshape(1, L * N, *act.shape[2:])
            # pred_h1 = trans_hx[1:].flatten(0, 1)
            pred_h0 = pred_h0.expand(wm.pred.num_layers, *pred_h0.shape)

            pred_hx, _ = wm.pred(pred_x, pred_h0.contiguous())
            pred_hx = pred_hx[-1]
            pred_loss = nn.functional.mse_loss(pred_hx, pred_h1)

            recons_rv = over_seq(wm.recons)(trans_hx)
            recons_loss = -recons_rv.log_prob(batch.obs).mean()

            # term_rv = over_seq(wm.term)(trans_hx)
            # term_loss1 = -term_rv[:-1].log_prob(0.0).mean()
            # term_loss2 = -term_rv[-1].log_prob(batch.term.float()).mean()
            # term_loss = term_loss1 + term_loss2
            term_rv = wm.term(trans_hx[-1])
            term_loss = -term_rv.log_prob(batch.term.float()).mean()

            rew_rv = over_seq(wm.rew)(trans_hx[1:])
            rew_loss = -rew_rv.log_prob(batch.reward).mean()

            wm_loss = pred_loss + recons_loss + term_loss + rew_loss
            wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            wm_opt.step()

            if should_log:
                for k in ["pred", "recons", "term", "rew", "wm"]:
                    board.add_scalar(f"train/{k}_loss", locals()[f"{k}_loss"])


if __name__ == "__main__":
    main()
