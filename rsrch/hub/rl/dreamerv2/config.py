from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, TypeAlias

from rsrch.utils.config import *

from . import env

ActType: TypeAlias = Literal["relu", "elu"]
NormType: TypeAlias = Literal["none", "batch", "layer"]


@dataclass
class RSSM:
    ensemble: int
    deter_size: int
    stoch: dict
    act: ActType
    norm: NormType
    hidden_size: int


@dataclass
class Encoders:
    obs: dict
    act: dict


@dataclass
class Decoders:
    obs: dict
    reward: dict
    term: dict


@dataclass
class WorldModel:
    rssm: RSSM
    encoders: Encoders
    decoders: Decoders
    opt: dict
    kl: dict
    reward_fn: Literal["same", "sign", "clip", "tanh"]
    clip_rew: tuple[float, float] | None
    coef: dict
    clip_grad: float | None


@dataclass
class ActorCritic:
    actor: dict
    critic: dict
    target_critic: dict | None
    actor_opt: dict
    critic_opt: dict
    horizon: int
    rew_norm: dict
    gamma: float
    gae_lambda: float
    actor_grad: Literal["dynamics", "reinforce", "both", "auto"]
    actor_grad_mix: float | dict
    actor_ent: float | dict
    clip_grad: float | None


@dataclass
class Dataset:
    capacity: int
    batch_size: int
    slice_len: int
    subseq_len: int | tuple[int, int] | None
    ongoing: bool
    prioritize_ends: bool


@dataclass
class Until:
    n: int
    of: str


@dataclass
class Every:
    n: int
    of: str
    iters: int


@dataclass
class Agent:
    expl_noise: float
    eval_noise: float


@dataclass
class Config:
    seed: int
    device: str
    dtype: Literal["float16", "float32"]
    env: env.Config
    wm: WorldModel
    ac: ActorCritic
    dataset: Dataset
    total: Until
    prefill: Until
    train_every: Every
    train_steps: int
    eval_every: Every
    agent: Agent
