from typing import TypeAlias

from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym

from .. import wm

State: TypeAlias = Tensor
StateDist: TypeAlias = D.Distribution


class Cell:
    def __init__(self, s: State) -> State:
        ...


class DeterWM(wm.WorldModel):
    obs_space: gym.Space
    obs_enc: wm.Encoder
    act_space: gym.Space
    act_enc: wm.Encoder
    act_dec: wm.ActDecoder
    prior: State
    deter_act_cell: Cell
    deter_obs_cell: Cell
    reward_pred: wm.Decoder
    term_pred: wm.Decoder

    def act_cell(self, prev_s: State, enc_act: Tensor):
        return D.Dirac(self.deter_act_cell(prev_s, enc_act), 1)

    def obs_cell(self, prev_s: State, enc_obs: Tensor):
        return D.Dirac(self.deter_obs_cell(prev_s, enc_obs), 1)
