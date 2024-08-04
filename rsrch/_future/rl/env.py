import concurrent.futures as cf
import io
import multiprocessing as mp
from abc import ABC, abstractmethod
from multiprocessing.reduction import AbstractReducer, ForkingPickler
from typing import Any, Callable, Iterable, Literal

import envpool
import gymnasium as gym
import numpy as np
from cloudpickle import CloudPickler

from rsrch.spaces.utils import from_gym


class Agent(ABC):
    def reset(self, obs):
        pass

    @abstractmethod
    def policy(self):
        ...

    def step(self, act, next_obs):
        pass


class Env(ABC):
    obs_space: Any
    act_space: Any

    @abstractmethod
    def reset(self) -> dict:
        ...

    @abstractmethod
    def step(self, act) -> tuple[dict, bool]:
        ...

    def rollout(self, agent: Agent):
        obs = None
        while True:
            if obs is None:
                obs = self.reset()
                agent.reset(obs)
                yield obs, False

            act = agent.policy()
            next_obs, final = self.step(act)
            agent.step(act, next_obs)
            yield {**next_obs, "act": act}, final

            obs = next_obs
            if final:
                obs = None


class VecAgent(ABC):
    def reset(self, idxes: np.ndarray, obs):
        pass

    @abstractmethod
    def policy(self, idxes: np.ndarray):
        ...

    def step(self, idxes: np.ndarray, act, next_obs):
        pass


class VecEnv(ABC):
    num_envs: int
    obs_space: Any
    act_space: Any

    @abstractmethod
    def rollout(self, agent: VecAgent) -> Iterable[tuple[int, dict, bool]]:
        ...


def unbind(x):
    if isinstance(x, dict):
        x = {k: unbind(v) for k, v in x.items()}
        n = len(next(iter(x.values())))
        return [{k: v[i] for k, v in x.items()} for i in range(n)]
    else:
        return x


class Envpool(VecEnv):
    """A vec env created via envpool."""

    def __init__(self, task_id: str, **kwargs):
        self.pool = envpool.make(task_id, env_type="gymnasium", **kwargs)
        self.obs_space = {"obs": from_gym(self.pool.observation_space)}
        self.act_space = from_gym(self.pool.action_space)
        self.num_envs = self.pool.config["num_envs"]
        self._batch_size = self.pool.config["batch_size"]
        self._actions = self.act_space.sample([self.num_envs])
        self._actions = self._actions.astype(np.int32)

    def rollout(self, agent: VecAgent):
        self.pool.async_reset()
        next_reset = [True for _ in range(self.num_envs)]

        while True:
            obs, reward, term, trunc, info = self.pool.recv()
            env_ids = info["env_id"].astype(np.int32)
            obs = {
                "obs": obs,
                "act": [self._actions[env_id] for env_id in env_ids],
                "reward": reward,
                "term": term,
                "trunc": trunc,
                **info,
            }
            all_obs = unbind(obs)

            reset_ids, reset_obs = [], []
            step_ids, step_acts, step_obs = [], [], []
            for idx in range(self._batch_size):
                env_id, env_obs = env_ids[idx], all_obs[idx]
                if next_reset[env_id]:
                    reset_ids.append(env_id)
                    reset_obs.append(env_obs)
                else:
                    step_ids.append(env_id)
                    step_acts.append(self._actions[env_id])
                    step_obs.append(env_obs)

            if len(step_ids) > 0:
                step_ids = np.array(step_ids)
                agent.step(step_ids, step_acts, step_obs)

            if len(reset_ids) > 0:
                reset_ids = np.array(reset_ids)
                agent.reset(reset_ids, reset_obs)

            for idx in range(self._batch_size):
                final = term[idx] or trunc[idx]
                yield env_ids[idx], (all_obs[idx], final)

            policy_ids = []
            for idx in range(self._batch_size):
                env_id = env_ids[idx]
                next_reset[env_id] = term[idx] or trunc[idx]
                if not next_reset[env_id]:
                    policy_ids.append(env_id)

            policy_ids = np.array(policy_ids)
            actions = agent.policy(policy_ids)
            for env_id, action in zip(policy_ids, actions):
                self._actions[env_id] = action

            self.pool.send({"action": self._actions[env_ids], "env_id": env_ids})


class GymEnv(Env):
    """An env created from a gymnasium.Env."""

    def __init__(self, env: gym.Env, seed=None, render=False):
        super().__init__()
        self.env = env
        self.render = render
        self.seed = seed
        self.obs_space = {"obs": from_gym(self.env.observation_space)}
        self.act_space = from_gym(self.env.action_space)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.seed = None
        res = {"obs": obs, "reward": 0.0, "term": False, "trunc": False, **info}
        if self.render:
            res["render"] = self.env.render()
        return res

    def step(self, act):
        next_obs, reward, term, trunc, info = self.env.step(act)
        res = {"obs": next_obs, "reward": reward, "term": term, "trunc": trunc, **info}
        if self.render:
            res["render"] = self.env.render()
        final = term or trunc
        return res, final


class CloudForkingPickler(ForkingPickler):
    @classmethod
    def dumps(cls, obj: Any, protocol: int | None = None) -> memoryview:
        buf = io.BytesIO()
        CloudPickler(buf, protocol).dump(obj)
        return buf.getbuffer()


class CloudReducer(AbstractReducer):
    ForkingPickler = CloudForkingPickler
    register = CloudForkingPickler.register


class ProcEnv(Env):
    """An env running in a subprocess."""

    def __init__(
        self,
        env_fn: Callable[[], Env],
        spawn_method: Literal["spawn", "fork"] = "spawn",
    ):
        self.recv_c, send_c = mp.Pipe(duplex=False)
        recv_c, self.send_c = mp.Pipe(duplex=False)

        ctx = mp.get_context(spawn_method)
        ctx.reducer = CloudReducer

        self.proc = ctx.Process(
            target=self.target,
            args=(env_fn, recv_c, send_c),
            daemon=True,
        )
        self.proc.start()

        self.obs_space, self.act_space = self.recv_c.recv()

    def reset(self):
        self.send_c.send(("reset",))
        return self.recv_c.recv()

    def step(self, act):
        self.send_c.send(("step", act))
        return self.recv_c.recv()

    def __del__(self):
        self.send_c.send(None)
        self.proc.join()

    @staticmethod
    def target(env_fn, recv_c, send_c):
        env: Env = env_fn()
        send_c.send((env.obs_space, env.act_space))

        while True:
            req = recv_c.recv()
            if req is None:
                break
            res = getattr(env, req[0])(*req[1:])
            send_c.send(res)


class EnvSet(VecEnv):
    """A vec env created from a list of regular envs."""

    def __init__(
        self,
        envs: list[Env],
        batch_size: int | None = None,
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self.batch_size = batch_size or self.num_envs
        self.obs_space = envs[0].obs_space
        self.act_space = envs[0].act_space

    def rollout(self, agent: VecAgent):
        with cf.ThreadPoolExecutor(self.batch_size) as pool:
            futures = {}
            actions = [None for _ in range(self.num_envs)]

            for idx in range(self.num_envs):
                fut = pool.submit(self.envs[idx].reset)
                futures[fut] = idx

            while True:
                counter = 0
                policy_idxes = []
                steps = []
                reset_idxes, reset_obs = [], []
                step_idxes, step_actions, step_obs = [], [], []

                while counter < self.batch_size:
                    done, _ = cf.wait(futures, return_when="FIRST_COMPLETED")
                    for fut in done:
                        env_idx = futures[fut]
                        del futures[fut]

                        if actions[env_idx] is None:
                            reset_idxes.append(env_idx)
                            obs = fut.result()
                            reset_obs.append(obs)
                            policy_idxes.append(env_idx)
                            steps.append((env_idx, (obs, False)))
                        else:
                            step_idxes.append(env_idx)
                            step_actions.append(actions[env_idx])
                            obs, final = fut.result()
                            step_obs.append(obs)
                            steps.append((env_idx, (obs, final)))
                            if final:
                                fut = pool.submit(self.envs[env_idx].reset)
                                futures[fut] = env_idx
                            else:
                                policy_idxes.append(env_idx)

                        counter += 1
                        if counter >= self.batch_size:
                            break

                if len(step_idxes) > 0:
                    step_idxes = np.array(step_idxes)
                    agent.step(step_idxes, step_actions, step_obs)

                if len(reset_idxes) > 0:
                    reset_idxes = np.array(reset_idxes)
                    agent.reset(reset_idxes, reset_obs)

                yield from steps

                policy_idxes = np.array(policy_idxes)
                policy = agent.policy(policy_idxes)
                for env_idx, action in zip(policy_idxes, policy):
                    fut = pool.submit(self.envs[env_idx].step, action)
                    futures[fut] = env_idx


class Pointwise(VecAgent):
    """A vec agent wrapper which (implicitly) applies a transform to each "sub-agent" of a vec agent (hence 'pointwise.')"""

    class Proxy(Agent):
        def __init__(self, parent: "Pointwise", env_idx: int):
            super().__init__()
            self.parent = parent
            self.env_idx = env_idx

        def reset(self, obs):
            self.parent._argv.append((obs,))

        def policy(self):
            return self.parent._policy[self.env_idx]

        def step(self, act, next_obs):
            self.parent._argv.append((act, next_obs))

    def __init__(self, agent: VecAgent, transform: Callable[[Agent], Agent]):
        super().__init__()
        self.agent = agent
        self.transform = transform
        self._agents: dict[int, Agent] = {}
        self._argv = []
        self._policy = {}

    def _make_proxy(self, env_idx: int):
        return self.transform(self.Proxy(self, env_idx))

    def reset(self, idxes: np.ndarray, obs):
        self._argv.clear()
        for env_idx, env_obs in zip(idxes, obs):
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            self._agents[env_idx].reset(env_obs)
        self.agent.reset(idxes, *zip(*self._argv))

    def policy(self, idxes: np.ndarray):
        actions = self.agent.policy(idxes)
        for env_idx, action in zip(idxes, actions):
            self._policy[env_idx] = action

        actions = []
        for env_idx in idxes:
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            action = self._agents[env_idx].policy()
            actions.append(action)

        return actions

    def step(self, idxes: np.ndarray, act, next_obs):
        self._argv.clear()
        for env_idx, env_act, env_obs in zip(idxes, act, next_obs):
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            self._agents[env_idx].step(env_act, env_obs)
        self.agent.step(idxes, *zip(*self._argv))
