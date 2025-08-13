from dataclasses import dataclass
from typing import Literal


@dataclass
class TimeDelta:
    n: int
    of: Literal["step", "epoch"]


@dataclass
class Config:
    # Base
    seed: int
    compute_dtype: Literal["float16", "bfloat16", "float32"]
    create_exp_commit: bool
    # Data
    dataset_path: str
    dataset_name: str | None
    tokenizer_path: str
    min_seq_len: int
    max_seq_len: int
    min_bucket_size: int
    min_bucket_char_count: int
    chars_per_batch: int
    # Model
    model_dim: int
    num_blocks: int
    num_heads: int
    hidden_dim: int
    # Training process
    sample_size: int
    sample_min_seq_len: int
    sample_max_seq_len: int
    val_every: TimeDelta
    max_completion_tokens: int


def main():
    import json
    from pathlib import Path

    from rsrch.utils.schema import get_schema

    with open(Path(__file__).parent / "config.schema.json", "w") as f:
        json.dump(get_schema(Config), f)


if __name__ == "__main__":
    main()
