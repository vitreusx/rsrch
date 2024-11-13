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


@torch.jit.script
def gae_only_ret(
    reward: Tensor,
    next_value: Tensor,
    next_gamma: Tensor,
    gae_lambda: float,
):
    ret = [reward[-1] + next_gamma[-1] * next_value[-1]]
    for t in range(len(reward) - 2, -1, -1):
        ret.append(
            reward[t]
            + next_gamma[t] * ((1 - gae_lambda) * next_value[t] + gae_lambda * ret[-1])
        )
    ret.reverse()
    return torch.stack(ret)
