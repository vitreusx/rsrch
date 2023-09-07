import numpy as np
from gymnasium.vector.utils import *
import torch
from ..spaces import *


def split_vec_info(info: dict, num_envs: int):
    split = [{} for _ in range(num_envs)]
    for k in info.keys():
        if k.startswith("_"):
            continue
        for i in range(num_envs):
            if f"_{k}" in info:
                if info[f"_{k}"][i]:
                    split[i][k] = info[k][i]
            else:
                split[i][k] = info[k]
    return split


def merge_vec_infos(infos: list[dict]):
    num_envs = len(infos)
    vec_info = {}
    for i, env_info in enumerate(infos):
        for k, v in env_info.items():
            if k not in vec_info:
                vec_info[k] = np.empty(num_envs, dtype=type(v))
                vec_info["_" + k] = np.zeros(num_envs, dtype=bool)
            vec_info[k][i] = env_info[k]
            vec_info["_" + k][i] = True
    return vec_info


@batch_space.register(Image)
def _(space: Image, n=1):
    spaces = []
    seeds = space.np_random.integers(0, int(1e8), size=n)
    for idx in range(n):
        copy = Image(space.shape, space._normalized, space._channels_last, seeds[idx])
        spaces.append(copy)
    return Tuple(spaces)


@batch_space.register(TensorImage)
def _(space: TensorImage, n=1):
    spaces = []
    seeds = torch.randint(0, int(1e8), (n,), generator=space.gen).cpu()
    for i in range(n):
        gen = torch.Generator(space.gen.device).manual_seed(seeds[i].item())
        copy = TensorImage(space.shape, space._normalized, space.device, gen)
        spaces.append(copy)

    return Tuple(spaces)


@batch_space.register(TensorBox)
def _(space: TensorBox, n=1):
    spaces = []
    seeds = torch.randint(0, int(1e8), (n,), generator=space.gen).cpu()
    for i in range(n):
        gen = torch.Generator(space.gen.device).manual_seed(seeds[i].item())
        copy = TensorBox(
            low=space.low.clone(),
            high=space.high.clone(),
            shape=space.shape,
            device=space.device,
            dtype=space.dtype,
            seed=gen,
        )
        spaces.append(copy)

    return Tuple(spaces)


@batch_space.register(TensorDiscrete)
def _(space: TensorDiscrete, n=1):
    spaces = []
    seeds = torch.randint(0, int(1e8), (n,), generator=space.gen).cpu()
    for i in range(n):
        gen = torch.Generator(space.gen.device).manual_seed(seeds[i].item())
        copy = TensorDiscrete(space.n, space.device, gen, space.start)
        spaces.append(copy)

    return Tuple(spaces)
