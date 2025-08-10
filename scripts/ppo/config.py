from dataclasses import dataclass
from typing import Literal

from rsrch.rl import sdk
from rsrch.rl.utils import polyak


@dataclass
class Config:
	seed: int
	device: str
	compute_dtype: Literal['float32', 'float16', 'bfloat16']
	create_exp_commit: bool

	env: sdk.Config
	num_train_envs: int
	num_val_envs: int | None
	steps_per_batch: int
	min_seq_len: int

	sample_every: int

	lr: float
	opt_eps: float
	update_epochs: int
	mb_size: int | None
	adv_norm: bool
	clip_coef: float
	clip_vloss: bool
	gamma: float
	gae_lambda: float
	clip_grad: float | None
	vf_coef: float
	rew_norm: Literal['id', 'clip', 'sign']
	ent_coef: float
	target_critic: polyak.Config | None
	share_encoder: bool
	target_critic: polyak.Config | None
	share_encoder: bool


def main():
	import json
	from pathlib import Path

	from rsrch.utils.schema import get_schema

	with open(Path(__file__).parent / 'config.schema.json', 'w') as f:
		json.dump(get_schema(Config), f)


if __name__ == '__main__':
	main()
