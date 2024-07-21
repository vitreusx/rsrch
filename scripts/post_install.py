import argparse
import subprocess

import tomli
from packaging.version import Version


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch", type=str, metavar="tag")
    p.add_argument("--opencv", type=str, choices=["full", "headless"])

    args = p.parse_args()

    with open("poetry.lock", "rb") as f:
        spec = tomli.load(f)

    if args.torch is not None:
        import torch
        import torchvision

        torch_v = Version(torch.__version__)
        torchvision_v = Version(torchvision.__version__)

        cmd = [
            "pip",
            "install",
            "-U",
            f"torch~={torch_v.base_version}",
            f"torchvision~={torchvision_v.base_version}",
            "--index-url",
            f"https://download.pytorch.org/whl/{args.torch}",
        ]
        subprocess.run(cmd)

    if args.opencv is not None:
        cmd = ["pip", "uninstall", "opencv-python", "opencv-python-headless"]
        subprocess.run(cmd)

        info = next(x for x in spec["package"] if x["name"] == "opencv-python")
        pkg = "opencv-python" if args.opencv == "full" else "opencv-python-headless"
        cmd = ["pip", "install", "-U", f"{pkg}~={info['version']}"]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
