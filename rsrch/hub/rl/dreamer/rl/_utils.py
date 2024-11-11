import torch
from torch import Tensor


@torch.jit.script
def gen_adv_est(
    reward: Tensor,
    value: Tensor,
    gamma: Tensor,
    gae_lambda: float,
):
    delta = (reward + gamma[1:] * value[1:]) - value[:-1]
    adv = [delta[-1]]
    for t in range(len(reward) - 1, 0, -1):
        adv.append(delta[t - 1] + gae_lambda * gamma[t] * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = value[:-1] + adv
    return adv, ret
