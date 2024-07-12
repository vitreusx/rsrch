from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, TypeAlias

from rsrch.utils.config import *

from . import env

ActType: TypeAlias = Literal["relu", "elu"]
NormType: TypeAlias = Literal["none", "batch", "layer"]


@dataclass
class Until:
    n: int
    of: str | None = None


@dataclass
class Every:
    n: int
    of: str | None = None
    iters: int | None = 1


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
class EarlyStop:
    margin: float
    patience: int
    min_steps: int


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
    opt: dict
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
    preload: str | None


@dataclass
class Agent:
    expl_noise: float
    eval_noise: float


@dataclass
class Debug:
    enabled: bool
    profile: bool
    record_memory: bool
    deterministic: bool
    detect_anomaly: bool

    def __post_init__(self):
        self.profile &= self.enabled
        self.record_memory &= self.enabled
        self.deterministic &= self.enabled
        self.detect_anomaly &= self.enabled


@dataclass
class Trainer:
    @dataclass
    class Basic:
        sched: Every

    @dataclass
    class Iterative:
        @dataclass
        class Slice:
            iterative: bool
            reset_every: int | None
            reset_coef: float
            opt_every: Every
            val_every: int
            stop_criteria: EarlyStop

        wm: Slice
        ac: Slice

    mode: Literal["basic", "iterative"]
    basic: Basic
    iterative: Iterative


@dataclass
class Sampler:
    env_mode: Literal["train", "val"]
    agent_mode: Literal["train", "val"]
    ckpt_path: str
    num_samples: int


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
    save_every: Every
    log_every: Every
    agent: Agent
    debug: Debug
    mode: Literal["train", "sample"]
    trainer: Trainer | None
    sampler: Sampler | None


def get_class(namespace: Any, name: str):
    """Try to acquire a class from a namespace. The name needs not be exactly
    the same - in particular, snake_case names are converted to CamelCase."""
    camel_case = "".join(x.capitalize() for x in name.split("_"))
    return getattr(namespace, camel_case)
