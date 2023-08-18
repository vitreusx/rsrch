import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions.transforms as T
from rsrch.types.tensorlike import Tensorlike

from . import dists as D


class State(Tensorlike):
    def __init__(self, stoch: Tensor, deter: Tensor):
        batch_shape = deter.shape[:-1]
        Tensorlike.__init__(self, batch_shape)

        self.stoch: Tensor
        self.register("stoch", stoch)

        self.deter: Tensor
        self.register("deter", deter)


class ActLayer(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._act = None
        if name == "relu":
            self._act = nn.ReLU(*args, **kwargs)
        elif name == "elu":
            self._act = nn.ELU(*args, **kwargs)

    def forward(self, x):
        if self._act is not None:
            x = self._act(x)
        return x


class NormLayer(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._norm = None
        if name == "batch1d":
            self._norm = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        if self._norm is not None:
            x = self._norm(x)
        return x


class NormalLayer(nn.Module):
    def __init__(self, in_features: int, stoch: int, std_act: str, min_std: float):
        super().__init__()
        self._std_act = std_act
        self._min_std = min_std
        self.fc = nn.Linear(in_features, 2 * stoch)

    def forward(self, x):
        x = self.fc(x)
        mean, std = x.chunk(2, -1)
        if self._std_act == "softplus":
            std = F.softplus(std)
        elif self._std_act == "sigmoid":
            std = F.sigmoid(std)
        elif self._std_act == "sigmoid2":
            std = 2.0 * F.sigmoid(std / 2)
        return D.Normal(mean, std + self._min_std)


class DiscreteLayer(nn.Module):
    def __init__(self, in_features: int, stoch: int, discrete: int):
        super().__init__()
        self._discrete = discrete
        self.fc = nn.Linear(in_features, stoch * discrete)

    def forward(self, x):
        x = self.fc(x)
        return D.MultiheadOHST(self._discrete, logits=x)


class DistLayer(nn.Module):
    def __init__(self, in_features, out_shape, name="mse", min_std=0.1, init_std=0.0):
        self._out_shape = out_shape
        self._name = name
        self._min_std = min_std
        self._init_std = init_std

        out_features = int(np.prod(out_shape))
        if name in ("normal", "tanh_normal", "trunc_normal"):
            out_features *= 2
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x).reshape(len(x), *self._out_shape)
        if self._name in ("normal", "tanh_normal", "trunc_normal"):
            mean, std = x.chunk(2, -1)
            if self._name == "normal":
                std = F.softplus(std)
                return D.Normal(mean, std, len(self._out_shape))
            elif self._name == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5.0)
                std = F.softplus(std + self._init_std) + self._min_std
                return D.TransformedDistribution(
                    D.Normal(mean, std, len(self._out_shape)),
                    T.TanhTransform(),
                )
            elif self._name == "trunc_normal":
                std = 2.0 * F.sigmoid((std + self._init_std) / 2.0) + self._min_std
                return D.TruncNormal(
                    mean, std, -1.0, 1.0, event_dims=len(self._out_shape)
                )
        elif self._name == "mse":
            return D.Normal(x, 1.0, len(self._out_shape))
        elif self._name == "binary":
            return D.Bernoulli(logits=x, event_dims=len(self._out_shape))
        elif self._name == "onehot":
            return D.OneHotCategoricalST(logits=x)
        else:
            raise NotImplementedError(self._name)


class EnsembleRSSM(nn.Module):
    def __init__(
        self,
        act_dim: int,
        embed_dim: int,
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete: bool | int = False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = act
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std

        self.img_in = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden),
            NormLayer(norm, hidden),
            ActLayer(act),
        )
        self._cell = nn.GRUCell(hidden, deter)

        self.img_out = nn.ModuleList([])
        for _ in range(ensemble):
            self.img_out.append(
                nn.Sequential(
                    nn.Linear(deter, hidden),
                    NormLayer(norm, hidden),
                    ActLayer(act),
                    self._stoch_layer(hidden),
                )
            )

        self.obs_out = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden),
            NormLayer(norm, hidden),
            ActLayer(act),
            self._stoch_layer(hidden),
        )

    def _stoch_layer(self, in_features: int):
        if self._discrete:
            return DiscreteLayer(
                in_features,
                self._stoch,
                self._discrete,
            )
        else:
            return NormalLayer(
                in_features,
                self._stoch,
                self._std_act,
                self._min_std,
            )

    @property
    def _proto(self):
        return next(self.parameters())

    def _cast(self, x: Tensor) -> Tensor:
        return x.type_as(self._proto)

    def initial(self, batch_size: int):
        args = dict(dtype=self._proto.dtype, device=self._proto.device)
        if self._discrete:
            stoch = torch.zeros([batch_size, self._stoch * self._discrete])
            logits = torch.zeros([batch_size, self._stoch * self._discrete], **args)
            stoch_rv = D.MultiheadOHST(self._discrete, logits=logits)
        else:
            stoch = torch.zeros([batch_size, self._stoch])
            loc = torch.zeros([batch_size, self._stoch], **args)
            scale = torch.zeros([batch_size, self._stoch], **args)
            stoch_rv = D.Normal(loc, scale, 1)

        deter = torch.zeros([batch_size, self._deter], **args)
        state = State(deter, stoch)

        return state, stoch_rv

    def observe(self, embed: Tensor, action: Tensor, is_first: Tensor, state=None):
        seq_len, batch_size = embed.shape[:2]
        if state is None:
            state = self.initial(batch_size)

        post, prior = [], []
        for step in range(seq_len):
            post_, prior_, state = self.obs_step(
                state,
                action[step],
                embed[step],
                is_first[step],
            )
            post.append(post_)
            prior.append(prior_)

        post, prior = torch.stack(post), torch.stack(prior)
        return post, prior

    def imagine(self, action: Tensor, state=None):
        seq_len, batch_size = action.shape[:2]
        if state is None:
            state = self.initial(batch_size)

        prior = []
        for step in range(seq_len):
            prior_, state = self.img_step(state, action[step])
            prior.append(prior_)

        prior = torch.stack(prior)
        return prior

    def obs_step(
        self,
        prev_state: State,
        prev_action: Tensor,
        embed: Tensor,
        is_first: Tensor,
        sample=True,
    ):
        if is_first.any():
            prev_state = prev_state.where(is_first, 0.0)
            prev_action = prev_action.where(is_first[..., None], 0.0)
        prior, img_state = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([img_state.deter, embed], -1)
        post = self.obs_out(x)
        stoch = post.rsample() if sample else post.mode()
        obs_state = State(stoch, img_state.deter)
        return post, prior, obs_state

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state.stoch)
        if self._discrete:
            prev_stoch = prev_stoch.flatten(-2, -1)
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self.img_in(x)
        deter = self._cell(x, prev_state.deter)
        prior = self._stoch_ensemble(deter)
        stoch = prior.rsample() if sample else prior.mode()
        img_state = State(stoch, deter)
        return prior, img_state

    def _stoch_ensemble(self, x):
        member_idx = torch.randint(0, self._ensemble, [])
        member = self.img_out[member_idx]
        return member(x)

    @staticmethod
    def kl_loss(
        self,
        post: D.Distribution,
        prior: D.Distribution,
        forward: bool,
        balance: float,
        free: float,
        free_avg: float,
    ):
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = D.kl_divergence(lhs, rhs)
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = D.kl_divergence(lhs, rhs.detach())
            value_rhs = D.kl_divergence(lhs.detach(), rhs)
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(nn.Module):
    def __init__(
        self,
        inputs: dict,
        act="elu",
        norm="none",
        conv_depth=48,
        conv_kernels=(4, 4, 4, 4),
        fc_layers=[400, 400, 400, 400],
    ):
        super().__init__()

        self._conv_keys, self._fc_keys = [], []
        self._conv_features, self._fc_features = 0, 0
        for key, (shape, type) in inputs.items():
            if type == "conv" and len(shape) == 3:
                self._conv_keys.append(key)
                self._conv_features += shape[0]
            elif type == "fc" and len(shape) == 1:
                self._fc_keys.append(key)
                self._fc_features += shape[0]

        if len(self._conv_keys) > 0:
            self.conv = nn.Sequential()
            in_features = self._conv_features
            for idx, k in enumerate(conv_kernels):
                out_features = (2**idx) * conv_depth
                self.conv.append(nn.Conv2d(in_features, out_features, k, 2))
                self.conv.append(NormLayer(norm, out_features))
                self.conv.append(ActLayer(act))
                in_features = out_features
            # self.conv.append(nn.AdaptiveAvgPool2d(1))
            self.conv.append(nn.Flatten())

        if len(self._fc_keys) > 0:
            self.fc = nn.Sequential()
            in_features = self._fc_features
            for idx, out_features in enumerate(fc_layers):
                self.fc.append(nn.Linear(in_features, out_features))
                self.fc.append(NormLayer(norm, out_features))
                self.fc.append(ActLayer(act))
                in_features = out_features

    def forward(self, input: dict) -> Tensor:
        outputs = []

        conv_x = {k: v for k, v in input.items() if k in self._conv_keys}
        if len(conv_x) > 0:
            conv_x = torch.cat([*conv_x.values()], 1)
            outputs.append(self.conv(conv_x))

        fc_x = {k: v for k, v in input.items() if k in self._fc_keys}
        if len(fc_x) > 0:
            fc_x = torch.cat([*fc_x.values()], 1)
            outputs.append(self.fc(fc_x))

        outputs = torch.cat(outputs, -1)
        return outputs


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        outputs: dict,
        act="elu",
        norm="none",
        conv_depth=48,
        conv_kernels=(4, 4, 4, 4),
        fc_layers=[400, 400, 400, 400],
    ):
        super().__init__()
        self._outputs = outputs
        self._conv_keys, self._fc_keys = [], []
        self._conv_splits, self._fc_splits = [], []
        for key, (shape, type) in outputs.items():
            if type == "conv" and len(shape) == 3:
                self._conv_keys.append(key)
                self._conv_splits.append(shape[0])
            elif type == "fc" and len(shape) == 1:
                self._fc_keys.append(key)
                self._fc_splits.append(shape[0])

        self._conv_features = sum(self._conv_splits)
        self._fc_features = sum(self._fc_splits)

        if len(self._conv_keys) > 0:
            self.conv = nn.Sequential()
            in_features = 2 ** (len(conv_kernels) + 1)
            self.conv.append(nn.Linear(input_dim, in_features))
            layer_idxes = reversed(range(len(conv_kernels)))
            for idx, k in zip(layer_idxes, conv_kernels):
                if idx > 0:
                    out_features = 2 ** (idx - 1) * conv_depth
                    convt = nn.ConvTranspose2d(in_features, out_features, k, 2)
                    self.conv.append(convt)
                    self.conv.append(NormLayer(norm, out_features))
                    self.conv.append(ActLayer(act))
                else:
                    out_features = self._conv_features
                    convt = nn.ConvTranspose2d(in_features, out_features, 2)
                    self.conv.append(convt)
                in_features = out_features

        if len(self._fc_keys) > 0:
            self.fc = nn.Sequential()
            in_features = input_dim
            for idx, out_features in enumerate(fc_layers):
                self.fc.append(nn.Linear(in_features, out_features))
                self.fc.append(NormLayer(norm))
                self.fc.append(ActLayer(act))
                in_features = out_features
            self.fc.append(nn.Linear(in_features, self._fc_features))

    def forward(self, input: Tensor) -> dict:
        outputs = {}

        if len(self._conv_keys) > 0:
            conv_x: Tensor = self.conv(input)
            conv_xs = conv_x.split_with_sizes(self._conv_splits)
            for conv_key, x in zip(self._conv_keys, conv_xs):
                outputs[conv_key] = D.Normal(x, 1.0, 3)

        if len(self._fc_keys) > 0:
            fc_x: Tensor = self.fc(input)
            fc_xs = fc_x.split_with_sizes(self._fc_splits)
            for fc_key, x in zip(self._fc_keys, fc_xs):
                outputs[fc_key] = D.Normal(x, 1.0, 1)

        return outputs


class FC(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_shape: torch.Size,
        hidden: list[int],
        norm="none",
        act="elu",
        dist={},
    ):
        out_shape = torch.Size(out_shape)
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(NormLayer(norm))
            layers.append(NormLayer(act))
            in_features = h
        layers.append(DistLayer(in_features, out_shape, **dist))
        super().__init__(*layers)
