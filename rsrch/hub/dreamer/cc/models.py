import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions.v3 as D
from rsrch.distributions.v3.tensorlike import Tensorlike


class State(Tensorlike):
    def __init__(self, stoch: Tensor, deter: Tensor):
        batch_size = stoch.shape[:-1]
        super().__init__(batch_size)

        self.stoch: Tensor
        self.register_field("stoch", stoch)

        self.deter: Tensor
        self.register_field("deter", deter)


class StateDist(Tensorlike):
    def __init__(self, stoch_rv: D.Distribution, deter: Tensor):
        batch_size = deter.shape[:-1]
        super().__init__(batch_size)

        self.stoch_rv: D.Distribution
        self.register_field("stoch_rv", stoch_rv)

        self.deter: Tensor
        self.register_field("deter", deter)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        stoch = self.stoch_rv.rsample(sample_shape)
        deter = self.deter.expand(*sample_shape, *self.deter.shape)
        return State(stoch, deter)


class RSSM(nn.Module):
    def __init__(
        self,
        act_dim: int,
        emb_dim: int,
        stoch=32,
        deter=200,
        hidden=200,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden

        self.img1 = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden),
            act_layer(),
        )
        self.cell = nn.GRUCell(hidden, deter)
        self.img2 = nn.Sequential(
            nn.Linear(deter, hidden),
            act_layer(),
        )
        self.img3 = nn.Linear(hidden, 2 * stoch)
        self.obs1 = nn.Sequential(nn.Linear(deter + emb_dim, hidden), act_layer())
        self.obs2 = nn.Linear(hidden, 2 * stoch)

    @property
    def dtype(self):
        return next(self._cell.parameters()).dtype

    def initial(self, batch_size: int):
        init = State(
            stoch=torch.zeros([self._stoch], dtype=self.dtype),
            deter=torch.zeros([self._deter], dtype=self.dtype),
        )
        return init.expand(batch_size, *init.shape)

    def observe(self, embed, action, state=None):
        num_steps, batch_size = action.shape[:2]
        if state is None:
            state = self.initial(batch_size)

        post_, prior_ = [], []
        for step in range(num_steps):
            post, prior = self.obs_step(state, action[step], embed[step])
            post_.append(post)
            prior_.append(prior)
            state = post.rsample()

        post, prior = torch.stack(post_), torch.stack(prior_)
        return post, prior

    def imagine(self, action, state=None):
        num_steps, batch_size = action.shape[:2]
        if state is None:
            state = self.initial(batch_size)

        prior_ = []
        for step in range(num_steps):
            prior = self.img_step(state, action[step])
            prior_.append(prior)
            state = prior.rsample()

        prior = torch.stack(prior_)
        return prior

    def obs_step(self, prev: State, act: Tensor, embed: Tensor):
        prior = self.img_step(prev, act)
        x = torch.cat([prior.deter, embed], -1)
        x = self.obs1(x)
        x: Tensor = self.obs2(x)
        mean, std = x.chunk(2, -1)
        std = nn.functional.softplus(std) + 0.1
        post = StateDist(D.Normal(mean, std, 1), prior.deter)
        return post, prior

    def img_step(self, prev: State, act: Tensor):
        x = torch.cat([prev.stoch, act], -1)
        x = self.img1(x)
        deter = self.cell(prev.deter, x)
        x = self.img2(deter)
        x: Tensor = self.img3(x)
        mean, std = x.chunk(2, -1)
        std = nn.functional.softplus(std) + 0.1
        prior = StateDist(D.Normal(mean, std, 1), deter)
        return prior


class ConvEncoder(nn.Sequential):
    def __init__(self, in_shape=(3, 64, 64), depth=32, act_layer=nn.ReLU):
        super().__init__(
            nn.Conv2d(in_shape[0], depth, 4, 2),
            act_layer(),
            nn.Conv2d(depth, 2 * depth, 4, 2),
            act_layer(),
            nn.Conv2d(2 * depth, 4 * depth, 4, 2),
            act_layer(),
            nn.Conv2d(4 * depth, 8 * depth, 4, 2),
            act_layer(),
            nn.Flatten(),
        )

        x = torch.empty(in_shape).unsqueeze(0)
        self.emb_dim = self(x).shape[-1]


class ConvDecoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        out_features=3,
        depth=32,
        act_layer=nn.ReLU,
    ):
        super().__init__()

        self.fc = nn.Linear(state_dim, 8 * depth)
        self.conv_net = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, 5, 2),
            act_layer(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, 2),
            act_layer(),
            nn.ConvTranspose2d(2 * depth, depth, 6, 2),
            act_layer(),
            nn.ConvTranspose2d(depth, out_features, 6, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.reshape(len(x), 1, 1, -1)
        x = self.conv_net(x)
        return D.Normal(x, 1.0, 3)


class DenseDecoder(nn.Module):
    def __init__(
        self, in_features, shape, layers, units, dist="normal", act_layer=nn.ELU
    ):
        super().__init__()
        self._shape = shape
        self._dist = dist

        self.fc = nn.Sequential()
        for layer in range(layers):
            self.fc.extend([nn.Linear(in_features, units), act_layer()])
            in_features = units
        self.fc.append(nn.Linear(in_features, int(np.prod(shape))))

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(len(x), *self._shape)
        if self._dist == "normal":
            return D.Normal(x, 1, len(self._shape))
        elif self._dist == "binary":
            return D.Bernoulli(logits=x, event_dims=len(self._shape))
        else:
            raise NotImplementedError(self._dist)


class ActionDecoder(nn.Module):
    def __init__(
        self,
        in_features,
        size,
        layers,
        units,
        dist="tanh_normal",
        act=nn.ELU,
        min_std=1e-4,
        init_std=5,
        mean_scale=5,
    ):
        super().__init__()
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._raw_init_std = np.log(np.exp(self._init_std) - 1)
        self._mean_scale = mean_scale

        self.fc = nn.Sequential()
        for _ in range(layers):
            self.fc.extend([nn.Linear(in_features, units), act()])
            in_features = units

        if dist == "tanh_normal":
            self.fc.append(nn.Linear(in_features, 2 * size))
        elif dist == "onehot":
            self.fc.append(nn.Linear(in_features, size))
        else:
            raise NotImplementedError(dist)

    def forward(self, x):
        param: Tensor = self.fc(x)
        if self._dist == "tanh_normal":
            mean, std = param.chunk(2, -1)
            mean = self._mean_scale * F.tanh(mean / self._mean_scale)
            std = F.softplus(std + self._raw_init_std) + self._min_std
            dist = D.SquashedNormal(mean, std, 1)
        else:
            dist = D.OneHotCategoricalST(logits=param)
        return dist
