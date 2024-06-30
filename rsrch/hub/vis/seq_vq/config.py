from rsrch.utils.config import *


@dataclass
class Dataset:
    slice_len: int
    batch_size: int
    ongoing: bool
    subseq_len: int | tuple[int, int] | None
    min_term_prob: float


@dataclass
class Config:
    device: str
    dtype: str
    seed: int
    samples: Path
    dataset: Dataset
    prefix_len: int
    test_every: int
