from . import config, rssm
from rsrch.rl.utils import make_env
from rsrch.rl import gym, data
from pathlib import Path


def get_frame_stack(env: gym.Env):
    while True:
        if isinstance(env, gym.wrappers.FrameStack):
            return env.num_stack
        elif isinstance(env, gym.Wrapper):
            env = env.env
        else:
            break


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    env_f = make_env.EnvFactory(cfg.env)
    env_spec = gym.EnvSpec(env_f.val_env())

    model = rssm.RSSM(env_spec, cfg.rssm)

    buffer = data.ChunkBuffer(
        nsteps=cfg.seq_len,
        capacity=cfg.buf_cap,
        stack_in=get_frame_stack(env_f.val_env()),
        persist=data.TensorStore(cfg.buf_cap),
    )


if __name__ == "__main__":
    main()
