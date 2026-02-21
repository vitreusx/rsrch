import json
from typing import Literal

from pydantic import BaseModel

from rsrch.rl import sdk
from rsrch.rl.utils import polyak


class Config(BaseModel):
    env: sdk.Config
    num_envs: int
    seed: int
    steps_per_batch: int
    min_seq_len: int
    adamw_lr: float
    adamw_eps: float
    # PPO config
    update_epochs: int
    mb_size: int | None
    adv_norm: bool
    clip_coef: float
    clip_vloss: bool
    gamma: float
    gae_lambda: float
    clip_grad: float | None
    vf_coef: float
    ent_coef: float
    target_critic: polyak.Config | None
    rew_transform: Literal["id", "clip", "sign"]


def main():
    schema = Config.model_json_schema()
    with open("configs/ppo_jax/schema.json", "w") as f:
        json.dump(schema, f)


if __name__ == "__main__":
    main()
