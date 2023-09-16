import torch
from torch import Tensor


def gae_adv_est(r: Tensor, v: Tensor, gamma: float, gae_lambda: float):
    delta = (r + gamma * v[1:]) - v[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(r))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = v[:-1] + adv
    return adv, ret
