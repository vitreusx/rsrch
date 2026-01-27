from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download_and_extract


class Args(BaseModel):
    source: str = "https://www.cs.toronto.edu/~kriz"
    """Source URL for the dataset."""
    data_root: str
    """Output directory in which to place the dataset."""


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    for name in ("cifar-10-python.tar.gz", "cifar-100-python.tar.gz"):
        source = urljoin(args.source, name)
        download_and_extract(
            url=source,
            dest=data_root,
            archive_dest=data_root / "archives" / name,
        )


if __name__ == "__main__":
    main()
