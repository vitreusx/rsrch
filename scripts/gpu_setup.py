import argparse
import shlex
import subprocess
from dataclasses import dataclass

from packaging.version import parse as vp


def _guess_torch_tag():
    from torch.utils.collect_env import get_env_info

    env_info = get_env_info()
    torch_v = vp(env_info.torch_version.split("+")[0])

    max_torch_v = "2.2.2"
    if torch_v > vp(max_torch_v):
        raise RuntimeError(f"Update the script for torch>={max_torch_v}")

    accel_type, arch_v = "cpu", None
    for accel_type_ in ["cuda"]:
        runtime_v = getattr(env_info, f"{accel_type_}_runtime_version", "N/A")
        if runtime_v != "N/A":
            accel_type = accel_type_
            arch_v = vp(runtime_v)
            break

    if accel_type == "cpu":
        arch_tag = "cpu"
    elif accel_type == "cuda":
        if torch_v >= vp("2.1") and arch_v >= vp("12.1"):
            arch_tag = "cu121"
        elif torch_v >= vp("2.0") and arch_v >= vp("11.8"):
            arch_tag = "cu118"
        elif vp("2.1") > torch_v >= vp("1.13") and arch_v >= vp("11.7"):
            arch_tag = "cu117"
        else:
            raise RuntimeError()

    return arch_tag


def get_torchvision(arch_tag: str):
    try:
        import torchvision
    except ImportError:
        ...

    torchvision_v = torchvision.__version__.split("+")[0]
    cmd = [
        "pip",
        "install",
        f"torchvision=={torchvision_v}+{arch_tag}",
        "--index-url",
        f"https://download.pytorch.org/whl/{arch_tag}",
    ]
    print(f"% {shlex.join(cmd)}")
    subprocess.run(cmd)


def get_torch(arch_tag=None):
    try:
        import torch

    except ImportError:
        return

    torch_v = torch.__version__.split("+")[0]
    cur_tag = [*torch.__version__.split("+"), "cpu"][1]

    if cur_tag == arch_tag:
        return

    if arch_tag is None:
        arch_tag = _guess_torch_tag()

    cmd = [
        "pip",
        "install",
        f"torch=={torch_v}+{arch_tag}",
        "--index-url",
        f"https://download.pytorch.org/whl/{arch_tag}",
    ]
    print(f"% {shlex.join(cmd)}")
    subprocess.run(cmd)

    get_torchvision(arch_tag)


@dataclass
class Args:
    arch_tag: str


def main():
    p = argparse.ArgumentParser(
        description="Install GPU-powered versions of packages.",
    )
    p.add_argument("--arch-tag", help="Explicit arch tag for Pytorch.")

    args = Args(**vars(p.parse_args()))

    get_torch(arch_tag=args.arch_tag)


if __name__ == "__main__":
    main()
