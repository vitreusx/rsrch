import torch
from torch import Tensor


def gen_adv_est(
    reward: Tensor,
    value: Tensor,
    gamma: float,
    gae_lambda: float,
):
    delta = (reward + gamma * value[1:]) - value[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(reward))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = value[:-1] + adv
    return adv, ret
