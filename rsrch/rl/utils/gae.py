import torch
from torch import Tensor


def gen_adv_est(
    reward: Tensor,
    value: Tensor,
    gamma: Tensor,
    gae_lambda: float,
):
    """Generalized Advantage Estimation (GAE).

    :param reward: Tensor :math:`r_{1:L}` of shape :math:`(L, N)` of rewards
    upon arriving at the state.
    :param value: Tensor :math:`v_{0:L}` of shape :math:`(L+1, N)` of value
    estimates for each state.
    :param gamma: Tensor :math:`\gamma_{0:L}` of shape :math:`(L+1, N)` of
    :math:`\gamma` discounts for each state. Usually, :math:`\gamma_t = \gamma`
    if state is non-terminal, and :math:`\gamma_t = 0` for terminal and
    post-terminal states.
    :param gae_lambda: GAE's :math:`\lambda` discount parameter.

    :return: A pair `(adv, ret)` of advantage and return estimates for all
        states except for the last one.
    """

    delta = (reward + gamma[1:] * value[1:]) - value[:-1]
    adv = [delta[-1]]
    for t in range(len(reward) - 1, 0, -1):
        adv.append(delta[t - 1] + gae_lambda * gamma[t] * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = value[:-1] + adv
    return adv, ret


def gae_only_ret(
    reward: Tensor,
    next_value: Tensor,
    next_gamma: Tensor,
    gae_lambda: float,
):
    """Variant of `gen_adv_est` computing only the returns.

    :param reward: Tensor :math:`r_{1:L}` of shape :math:`(L, N)` of rewards
    upon arriving at the state.
    :param next_value: Tensor :math:`r_{1:L}` of shape :math:`(L, N)` of
    next-value estimates.
    :param next_gamma: Tensor :math:`\gamma_{1:L}` of shape :math:`(L, N)` of
    :math:`\gamma` discounts for each state aside from the first one.
    """

    ret = [reward[-1] + next_gamma[-1] * next_value[-1]]
    for t in range(len(reward) - 2, -1, -1):
        ret.append(
            reward[t]
            + next_gamma[t] * ((1 - gae_lambda) * next_value[t] + gae_lambda * ret[-1])
        )
    ret.reverse()
    return torch.stack(ret)


gen_adv_est = torch.jit.script(gen_adv_est)
gae_only_ret = torch.jit.script(gae_only_ret)
