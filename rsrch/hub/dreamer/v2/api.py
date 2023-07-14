from typing import Generic, Protocol, TypeVar

import torch.distributions as D
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ObsEncoder(Protocol, Generic[ObsType]):
    def __call__(self, obs: ObsType) -> Tensor:
        ...


class ActEncoder(Protocol, Generic[ActType]):
    def __call__(self, act: ActType) -> Tensor:
        ...


class ActDecoder(Protocol, Generic[ActType]):
    def __call__(self, act: Tensor) -> ActType:
        ...


class RecurModel(Protocol):
    def __call__(self, cur_h: Tensor, cur_z: Tensor, enc_act: Tensor) -> Tensor:
        ...


class TransPred(Protocol):
    def __call__(self, cur_h: Tensor) -> D.Distribution:
        ...


class ReprModel(Protocol):
    def __call__(self, cur_h: Tensor, enc_obs: Tensor) -> D.Distribution:
        ...


class VarPred(Protocol):
    def __call__(self, cur_h: Tensor, cur_z: Tensor) -> D.Distribution:
        ...


class RSSM(Protocol):
    prior_h: Tensor
    prior_z: Tensor
    obs_enc: ObsEncoder
    act_enc: ActEncoder
    act_dec: ActDecoder
    recur_model: RecurModel
    repr_model: ReprModel
    trans_pred: TransPred
    rew_pred: VarPred
    term_pred: VarPred


class Critic(Protocol):
    def __call__(self, cur_h: Tensor, cur_z: Tensor) -> Tensor:
        ...


class Actor(Protocol):
    def __call__(self, cur_h: Tensor, cur_z: Tensor) -> D.Distribution:
        ...
