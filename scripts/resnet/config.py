from dataclasses import dataclass
from typing import Literal


@dataclass
class TimeDelta:
    n: int
    of: Literal["step", "epoch"]


@dataclass
class Config:
    seed: int
    profile: bool
    create_exp_commit: bool
    compute_dtype: Literal["float16", "bfloat16", "float32"]
    batch_size: int
    val_batch_size: int | None
    train_for: TimeDelta
    log_every: TimeDelta
    val_every: TimeDelta
    max_val_samples: int | None
    save_every: TimeDelta | None


def main():
    import json
    from pathlib import Path

    from rsrch.utils.schema import get_schema

    with open(Path(__file__).parent / "config.schema.json", "w") as f:
        json.dump(get_schema(Config), f)


if __name__ == "__main__":
    main()
