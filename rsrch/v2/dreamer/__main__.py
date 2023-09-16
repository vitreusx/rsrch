import numpy as np
import torch
from torch import Tensor
from rsrch.rl.data.core import ChunkBatch

from rsrch.rl.gym.vector.events import VecReset, VecStep
from . import config, rssm, wm
from rsrch.rl.utils.make_env import EnvFactory
from rsrch.rl import gym, data
from rsrch.rl.data import rollout
from pathlib import Path
from rsrch.exp.board.wandb import Wandb
from rsrch.exp.vcs import WandbVCS
from rsrch.utils import cron
from rsrch.exp.pbar import ProgressBar


T = gym.wrappers.transforms


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)

    class EnvLoader:
        def __init__(self):
            self._env_f = EnvFactory(cfg.env, record_stats=True, to_tensor=False)

        def base_env(self):
            return self._env_f.base_env()

        def val_env(self):
            env = self.base_env()
            env = gym.wrappers.ToTensor(env)
            if cfg.env.stack:
                env = gym.wrappers.FrameStack2(env, cfg.env.stack)
                env = gym.wrappers.Apply(env, T.Concat(env.observation_space, 0))
            return env

        def exp_env(self):
            env = self.base_env()
            if isinstance(env.observation_space, gym.spaces.Image):
                obs_t = T.ToTensorImage(env.observation_space, normalize=False)
            elif isinstance(env.observation_space, gym.spaces.Box):
                obs_t = T.ToTensorBox(env.observation_space)
            env = gym.wrappers.Apply(env, obs_t, T.ToTensor(env.action_space).inv)
            if cfg.env.stack:
                env = gym.wrappers.FrameStack2(env, cfg.env.stack)
            return env

    loader = EnvLoader()
    val_spec = gym.EnvSpec(loader.val_env())
    exp_spec = gym.EnvSpec(loader.exp_env())

    def exp_to_val(obs: list[tuple[Tensor]]):
        obs = torch.stack([torch.cat(o) for o in obs])
        if isinstance(val_spec.observation_space, gym.spaces.TensorImage):
            obs = obs / 255.0
        return obs

    nets = rssm.AllNets(val_spec, cfg.rssm)
    nets = nets.to(device)

    if cfg.exp_envs > 1:
        exp_env = gym.vector.AsyncVectorEnv2(
            [loader.exp_env] * cfg.exp_envs,
            num_workers=cfg.env_workers,
        )
    else:
        exp_env = gym.vector.SyncVectorEnv([loader.exp_env])

    val_env = gym.vector.AsyncVectorEnv2(
        env_fns=[loader.val_env] * cfg.val_episodes,
        num_workers=cfg.env_workers,
    )

    class Agent(gym.vector.AgentWrapper):
        def __init__(self, type="exp"):
            self._type = type
            env = {"exp": exp_env, "val": val_env}[type]
            super().__init__(wm.VecEnvAgent(nets.wm, nets.actor, env, device))

        def reset(self, data: VecReset):
            if self._type == "exp":
                data.obs = exp_to_val(data.obs)
            data.obs = torch.stack([*data.obs])
            return super().reset(data)

        def observe(self, data: VecStep):
            if self._type == "exp":
                data.next_obs = exp_to_val(data.next_obs)
            data.next_obs = torch.stack([*data.next_obs])
            return super().observe(data)

    exp_agent, val_agent = Agent("exp"), Agent("val")

    sampler = data.UniformSampler()

    buffer = data.ChunkBuffer(
        nsteps=cfg.seq_len,
        capacity=cfg.buf_cap,
        stack_in=cfg.env.stack,
        store=data.TensorStore(cfg.buf_cap),
        sampler=sampler,
    )

    exp_iter = iter(rollout.steps(exp_env, exp_agent))
    ep_id = [None for _ in range(exp_env.num_envs)]

    env_step = 0
    should_val = cron.Every(lambda: env_step, cfg.val_every)

    pbar = ProgressBar(total=cfg.total_steps)
    board = Wandb(project="dreamer")
    vcs = WandbVCS()
    vcs.save()

    def val_epoch():
        val_returns = []
        val_eps = rollout.episodes(val_env, val_agent, max_episodes=cfg.val_episodes)
        for _, ep in val_eps:
            val_returns.append(sum(ep.reward))
        board.add_scalar("val/returns", np.mean(val_returns), step=env_step)

    def train_step():
        nonlocal env_step
        env_idx, step = next(exp_iter)
        ep_id[env_idx], _ = buffer.push(ep_id[env_idx], step)
        env_step += 1
        pbar.update()

        if len(buffer) > cfg.prefill:
            idxes = sampler.sample(cfg.batch_size)
            batch = buffer[idxes]

            bs, seq_len = len(batch), len(batch.act)
            obs, act, rew = [], [], []
            for seq_i in range(seq_len + 1):
                for seq in batch:
                    obs.extend(seq.obs[seq_i])
                    if seq_i > 0:
                        act.append(seq.act[seq_i - 1])
                        rew.append(seq.reward[seq_i - 1])

            obs_shape = exp_spec.observation_space.shape
            obs = torch.stack(obs).reshape(bs, seq_len + 1, -1, obs_shape[1:])
            act_shape = exp_spec.action_space.shape
            act = torch.stack(act).reshape(bs, seq_len, *act_shape)
            rew = torch.tensor(rew)
            term = torch.tensor([seq.term for seq in batch])
            batch = ChunkBatch(obs, act, rew, term)

            if isinstance(exp_spec.observation_space, gym.spaces.TensorImage):
                batch.obs = batch.obs / 255.0

            batch = batch.to(device)

            ...

    while True:
        if should_val:
            val_epoch()
        if env_step > cfg.total_steps:
            break
        train_step()


if __name__ == "__main__":
    main()
