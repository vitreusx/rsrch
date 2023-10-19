from .utils import *
from .config import Config
import torch
from torch import Tensor
from rsrch.rl import data
import envpool


class Loader:
    """A utility class for creating environments and processing attendant data."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._infer_specs()

    def _infer_specs(self):
        env = self._base_env()

        self.visual = False
        if isinstance(env.observation_space, gym.spaces.Box):
            self.visual = len(env.observation_space.shape) > 1
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)

        env = self.exp_env()

        obs = vec_utils.create_empty_array(env.observation_space, 1)
        net_obs = self._load_obs(obs)[0]
        if self.visual:
            self.obs_space = gym.spaces.TensorImage(net_obs.shape, net_obs.dtype)
        else:
            low = self._load_obs(env.observation_space.low[None])[0]
            high = self._load_obs(env.observation_space.high[None])[0]
            self.obs_space = gym.spaces.TensorBox(
                net_obs.shape, low, high, dtype=net_obs.dtype
            )

        if self.discrete:
            self.act_space = gym.spaces.TensorDiscrete(env.action_space.n)
        else:
            act = vec_utils.create_empty_array(env.action_space, 1)
            net_act = self._load_act(act)[0]
            low = self._load_act(env.action_space.low[None])[0]
            high = self._load_act(env.action_space.high[None])[0]
            self.act_space = gym.spaces.TensorBox(
                net_act.shape, low, high, dtype=net_act.dtype
            )

    def _base_env(self):
        if self.cfg.type == "atari":
            env = self._atari_env()
        else:
            env = gym.make(self.cfg.id)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        return env

    def _atari_env(self):
        atari = self.cfg.atari
        env = gym.make(self.cfg.id, frameskip=1)

        env = gym.wrappers.AtariPreprocessing(
            env=env,
            frame_skip=atari.frame_skip,
            screen_size=atari.screen_size,
            terminal_on_life_loss=False,
            grayscale_obs=atari.grayscale,
            grayscale_newaxis=True,
            scale_obs=False,
            noop_max=atari.noop_max,
        )

        env = BoxTransform(env, lambda x: np.transpose(x, (2, 0, 1)))

        if atari.fire_reset:
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = gym.wrappers.FireResetEnv(env)

        return env

    def _atari_envpool(self, n: int, val=False):
        task_id = self.cfg.id
        if task_id.startswith("ALE/"):
            task_id = task_id[len("ALE/") :]

        task_id, task_version = task_id.split("-")
        if task_version not in ("v4", "v5"):
            return

        if task_id.endswith("NoFrameskip"):
            task_id = task_id[: -len("NoFrameskip")]

        if f"{task_id}-v5" not in envpool.list_all_envs():
            return

        atari = self.cfg.atari

        max_steps = (self.cfg.time_limit or int(1e6)) // atari.frame_skip

        env = envpool.make(
            task_id=f"{task_id}-v5",
            env_type="gymnasium",
            num_envs=n,
            max_episode_steps=max_steps,
            img_height=atari.screen_size,
            img_width=atari.screen_size,
            stack_num=self.cfg.stack,
            gray_scale=atari.grayscale,
            frame_skip=atari.frame_skip,
            noop_max=atari.noop_max,
            episodic_life=atari.episodic_life and not val,
            zero_discount_on_life_loss=False,
            reward_clip=False,
            repeat_action_probability={"v5": 0.25, "v4": 0.0}[task_version],
            use_inter_area_resize=True,
            use_fire_reset=atari.fire_reset,
            full_action_space=False,
        )
        env = VecEnvPool(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Envpool concatenates, instead of stacking. This fixes this behavior.
        f = lambda x: x.reshape([len(x), self.cfg.stack, -1, *x.shape[2:]])
        env = BoxTransformV(env, f)

        return env

    def val_env(self):
        """Enviroment used during validation."""
        env = self._base_env()

        if self.cfg.stack is not None:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)

        return env

    def val_envs(self, num_envs: int):
        """Vectorized `val_env`."""
        if self.cfg.type == "atari":
            env = self._atari_envpool(num_envs, val=True)

        if num_envs == 1:
            return gym.vector.SyncVectorEnv([self.val_env])
        else:
            return gym.vector.AsyncVectorEnv([self.val_env] * num_envs)

    def exp_env(self):
        """Environment used for experience collection. Includes things such
        as reward clipping, episodic lives etc."""

        env = self._base_env()

        if self.cfg.type == "atari":
            atari = self.cfg.atari
            if atari.episodic_life:
                env = gym.wrappers.EpisodicLifeEnv(env)

        env = self._reward_t(env)

        if self.cfg.stack is not None:
            env = gym.wrappers.FrameStack(env, self.cfg.stack)

        return env

    def _reward_t(self, env: gym.Env | gym.VectorEnv):
        if self.cfg.reward not in (None, "keep"):
            if self.cfg.reward == "sign":
                rew_f = np.sign
                reward_range = (-1, 1)
            elif isinstance(self.cfg.reward, tuple):
                r_min, r_max = self.cfg.reward
                rew_f = lambda r: np.clip(r, r_min, r_max)
                reward_range = (r_min, r_max)

            if gym.is_vector_env(env):
                env = gym.vector.wrappers.TransformReward(env, rew_f)
            else:
                env = gym.wrappers.TransformReward(env, rew_f)
            env.reward_range = reward_range
        return env

    def exp_envs(self, num_envs: int):
        """Vectorized `exp_env`."""

        if self.cfg.type == "atari":
            env = self._atari_envpool(num_envs, val=False)
            if env is not None:
                env = self._reward_t(env)
                return env

        if num_envs == 1:
            return gym.vector.SyncVectorEnv([self.exp_env])
        else:
            return gym.vector.AsyncVectorEnv([self.exp_env] * num_envs)

    def make_step_buffer(self, capacity: int, sampler=None):
        """Create a step buffer."""

        pre_stack_env = self._base_env()

        return data.ChunkBuffer(
            num_steps=1,
            num_stack=self.cfg.stack,
            capacity=capacity,
            obs_space=pre_stack_env.observation_space,
            act_space=pre_stack_env.action_space,
            sampler=sampler,
            store=data.NumpySeqStore(
                capacity=capacity,
                obs_space=pre_stack_env.observation_space,
                act_space=pre_stack_env.action_space,
            ),
        )

    def fetch_step_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Given a step buffer and a list of idxes, fetch a step batch and
        transform it to tensor form. The observations are properly stacked."""

        batch = buffer[idxes]

        obs = np.stack([seq.obs[0] for seq in batch])
        obs = self._load_obs(obs)

        next_obs = np.stack([seq.obs[1] for seq in batch])
        next_obs = self._load_obs(next_obs)

        act = np.stack([seq.act[0] for seq in batch])
        act = self._load_act(act)

        reward = np.stack([seq.reward[0] for seq in batch])
        reward = torch.as_tensor(reward, dtype=obs.dtype)

        term = torch.as_tensor([seq.term for seq in batch])
        term = torch.as_tensor(term)

        return data.TensorStepBatch(obs, act, next_obs, reward, term)

    def make_chunk_buffer(self, capacity: int, num_steps: int, sampler=None):
        """Create a chunk buffer."""

        pre_stack_env = self._base_env()

        return data.ChunkBuffer(
            num_steps=num_steps,
            num_stack=self.cfg.stack,
            capacity=capacity,
            obs_space=pre_stack_env.observation_space,
            act_space=pre_stack_env.action_space,
            sampler=sampler,
            store=data.NumpySeqStore(
                capacity=capacity,
                obs_space=pre_stack_env.observation_space,
                act_space=pre_stack_env.action_space,
            ),
        )

    def fetch_chunk_batch(self, buffer: data.ChunkBuffer, idxes: np.ndarray):
        """Fetch a step batch and transform it to tensor form. The observations
        are stacked as if using FrameStack."""

        batch = buffer[idxes]
        seq_len, batch_size = len(batch[0].act), len(batch)

        obs = np.stack([seq.obs for seq in batch], axis=1)
        if self.cfg.stack is not None:
            obs = np_flatten(obs, 2, 3)
        obs = self._load_obs(obs.reshape(-1, *obs.shape[2:]))
        obs = obs.reshape([seq_len + 1, batch_size, *obs.shape[1:]])

        act = np.stack([seq.act for seq in batch], axis=1)
        act = self._load_act(act.reshape(-1, *act.shape[2:]))
        act = act.reshape(seq_len, batch_size, *act.shape[1:])

        reward = np.stack([seq.reward for seq in batch], axis=1)
        reward = torch.as_tensor(reward, dtype=obs.dtype)

        term = torch.as_tensor([seq.term for seq in batch])

        return data.ChunkBatch(obs, act, reward, term)

    @property
    def VecAgent(loader):
        """A wrapper for VectorAgent that matches the input format."""

        class _Agent(gym.vector.AgentWrapper):
            def __init__(self, agent: gym.vector.Agent):
                super().__init__(agent)

            def reset(self, idxes, obs, info):
                obs = loader._load_obs(obs)
                return super().reset(idxes, obs, info)

            def observe(self, idxes, next_obs, term, trunc, info):
                next_obs = loader._load_obs(next_obs)
                return super().observe(idxes, next_obs, term, trunc, info)

            def policy(self, obs):
                obs = loader._load_obs(obs)
                return super().policy(obs)

            def step(self, act):
                act = loader._load_act(act)
                return super().step(act)

        return _Agent

    def _load_obs(self, obs: np.ndarray) -> Tensor:
        """Load a batch of observations."""
        obs: Tensor = torch.as_tensor(obs)
        if self.cfg.stack is not None:
            obs = obs.flatten(1, 2)
        if self.visual:
            if obs.dtype == torch.uint8:
                obs = obs / 255.0
        else:
            obs = obs.to(torch.float32)
        return obs

    def _load_act(self, act: np.ndarray) -> Tensor:
        """Convert a batch of actions."""
        act = torch.as_tensor(act)
        if torch.is_floating_point(act):
            act = act.to(torch.float32)
        return act
