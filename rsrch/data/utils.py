import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm


def download(
    url: str,
    dest: str | Path,
    display_pbar: bool = True,
):
    dest = Path(dest)
    if dest.exists():
        return

    scheme = urlparse(url).scheme
    if scheme in ("http", "https"):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(dest, "wb") as file,
            tqdm(
                desc=f"{url} -> {dest}",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                disable=not display_pbar,
            ) as bar,
        ):
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        msg = f"Unsupported scheme {scheme}"
        raise ValueError(msg)


def download_and_extract(
    url: str,
    dest: str | Path,
    archive_dest: str | Path | None = None,
    display_pbar: bool = True,
):
    with ExitStack() as stack:
        if archive_dest is None:
            tmp_d = stack.enter_context(tempfile.TemporaryDirectory())
            archive_dest = Path(tmp_d) / PurePosixPath(urlparse(url).path).name
        archive_dest = Path(archive_dest)

        download(
            url=url,
            dest=archive_dest,
            display_pbar=display_pbar,
        )

        archive_fmt = PurePosixPath(urlparse(url).path).suffix
        if archive_fmt == ".zip":
            archive = zipfile.ZipFile(archive_dest)
        elif archive_fmt == ".tar":
            archive = tarfile.open(archive_dest, "r")  # noqa: SIM115
        elif archive_fmt == ".tar.gz":
            archive = tarfile.open(archive_dest, "r:gz")  # noqa: SIM115
        else:
            msg = f"Unknown archive format {archive_fmt}"
            raise ValueError(msg)

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        with archive as xf:
            xf.extractall(dest)  # noqa: S202
