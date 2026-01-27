from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download


class Args(BaseModel):
    source: str = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    """URL for the nyu_depth_v2_labeled.mat file."""
    data_root: str
    """Output directory in which to place the dataset."""


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    download(
        url=args.source,
        dest=data_root / "nyu_depth_v2_labeled.mat",
    )


if __name__ == "__main__":
    main()
