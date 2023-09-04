from dataclasses import dataclass


@dataclass
class Opt:
    name: str
    lr: float
    eps: float


@dataclass
class Config:
    @dataclass
    class Buffer:
        capacity: int
        prefill: int

    @dataclass
    class Sched:
        @dataclass
        class Value:
            opt_every: int
            opt_iters: int

        Actor = Value

        total_steps: int
        env_batch: int
        opt_batch: int
        value: Value
        actor: Actor

    @dataclass
    class Infra:
        device: str
        env_workers: int

    @dataclass
    class Env:
        @dataclass
        class Atari:
            screen_size: int
            frame_skip: int
            term_on_life_loss: bool
            grayscale: bool
            noop_max: int

        name: str
        atari: Atari
        reward: str | tuple[int, int]
        time_limit: int

    @dataclass
    class Actor:
        memory: int
        opt: Opt

    @dataclass
    class Value:
        @dataclass
        class Target:
            sync_every: int
            tau: float

        target: Target
        opt: Opt

    @dataclass
    class Alpha:
        autotune: bool
        ent_scale: float
        value: float | None
        opt: Opt

    @dataclass
    class Exp:
        val_every: int
        val_episodes: int
        val_envs: int
        log_every: int

    @dataclass
    class Profile:
        enabled: bool
        wait: int
        warmup: int
        active: int
        repeat: int
        export_trace: bool
        export_stack: bool

    buffer: Buffer
    sched: Sched
    infra: Infra
    env: Env
    actor: Actor
    value: Value
    alpha: Alpha
    gamma: float
    exp: Exp
    profile: Profile
