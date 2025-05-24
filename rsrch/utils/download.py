import io
from pathlib import Path
from typing import Literal

import requests
from tqdm.auto import tqdm

from .path import sanitize


def download_url(
    url: str,
    cache: bool = True,
    cache_dir: str = "~/.cache/rsrch",
    mode: Literal["r", "rb"] = "rb",
    progress_bar: bool = True,
):
    if cache:
        cache_dir = Path(cache_dir).expanduser()
        url_as_path = sanitize(url)
        dest = cache_dir / url_as_path
        if dest.exists():
            return open(dest, mode)

    resp = requests.get(url, stream=True)
    length = int(resp.headers["Content-Length"])
    if cache:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as dest_f:
            chunk_size = 1024**2
            with tqdm(total=length, disable=not progress_bar) as pbar:
                for chunk in resp.iter_content(chunk_size):
                    dest_f.write(chunk)
                    pbar.update(len(chunk))

        return open(dest, mode)
    else:
        return io.BytesIO(resp.content)
