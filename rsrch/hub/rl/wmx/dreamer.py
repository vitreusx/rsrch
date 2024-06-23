from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import *
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.types import Tensorlike
from rsrch.utils.preview import make_grid

from . import data
from .tokenizer import InputSpec, Tokenizer
from .utils import over_seq, pass_gradient


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)


class StateDist(D.Distribution, Tensorlike):
    def __init__(
        self,
        *,
        deter: Tensor,
        logits: Tensor,
        num_tokens: int,
        vocab_size: int,
    ):
        logits = logits.reshape(*logits.shape[:-1], num_tokens, vocab_size)
        ind_rv = D.Categorical(logits=logits)
        Tensorlike.__init__(self, ind_rv.shape[:-1])
        self.num_tokens, self.vocab_size = num_tokens, vocab_size
        self.event_shape = tuple([self.num_tokens])
        self.deter = self.register("deter", deter)
        self.ind_rv = self.register("ind_rv", ind_rv)

    def sample(self, sample_shape=()):
        deter = self.deter.expand(*sample_shape, *self.deter.shape)
        idx = self.ind_rv.sample(sample_shape)
        stoch = F.one_hot(idx, self.vocab_size).flatten(-2)
        stoch = stoch.type_as(self.ind_rv.logits)
        return State(deter, stoch)

    def rsample(self, sample_shape=()):
        deter = self.deter.expand(*sample_shape, *self.deter.shape)
        idx = self.ind_rv.sample(sample_shape)
        stoch = F.one_hot(idx, self.vocab_size)
        stoch = stoch.type_as(self.ind_rv.logits)
        stoch = pass_gradient(stoch, self.ind_rv.logits)
        stoch = stoch.flatten(-2)
        return State(deter, stoch)

    def entropy(self):
        return sum_rightmost(self.ind_rv.entropy(), 1)


@D.register_kl(StateDist, StateDist)
def _(p: StateDist, q: StateDist):
    return D.kl_divergence(p.ind_rv, q.ind_rv)


class WorldModel(nn.Module):
    def __init__(self, input_spec: InputSpec):
        super().__init__()
        self.spec = input_spec

        self.tag_size = 64
        self._tag_emb = nn.Embedding(2, self.tag_size)

        self.obs_features = 64
        self._obs_emb = nn.Embedding(self.spec.obs.vocab_size, self.obs_features)
        obs_size = self.spec.obs.num_tokens * self.obs_features

        self.act_features = 64
        self._act_emb = nn.Embedding(self.spec.act.vocab_size, self.act_features)
        act_size = self.spec.act.num_tokens * self.act_features

        self.act_idxes = slice(0, act_size)
        self.obs_idxes = slice(act_size, act_size + obs_size)
        self.input_size = self.obs_idxes.stop

        self.seq_hidden, self.seq_layers = 1024, 2
        self.deter_dim = self.seq_hidden
        self.infer_rnn = nn.GRU(self.input_size, self.seq_hidden, self.seq_layers)

        self.stoch_tokens, self.stoch_vocab_size = 32, 32
        stoch_features = self.stoch_vocab_size
        self.stoch_dim = self.state_dim = self.stoch_tokens * stoch_features

        self.deter_cell = nn.GRUCell(self.stoch_dim + act_size, self.seq_hidden)

        stoch_logits = self.stoch_tokens * self.stoch_vocab_size
        self._stoch_fc = nn.Sequential(
            nn.LayerNorm(self.deter_dim),
            nn.Linear(self.deter_dim, self.deter_dim),
            nn.ReLU(),
            nn.LayerNorm(self.deter_dim),
            nn.Linear(self.deter_dim, stoch_logits),
        )

        obs_logits = input_spec.obs.num_tokens * input_spec.obs.vocab_size
        self._obs_dec = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, obs_logits),
        )

    def obs_emb(self, obs: Tensor) -> Tensor:
        return self._obs_emb(obs).flatten(-2)

    def act_emb(self, act: Tensor) -> Tensor:
        return self._act_emb(act).flatten(-2)

    def pack_infer(
        self,
        obs: Tensor,
        act: Tensor,
        *,
        obs_first=True,
        encoded=False,
    ):
        if encoded:
            enc_obs, enc_act = obs, act
        else:
            enc_obs = self.obs_emb(obs)
            enc_act = self.act_emb(act)

        seq_len, batch_size = enc_obs.shape[:2]
        seq_shape = (seq_len, batch_size, self.input_size)
        seq_x = torch.zeros(seq_shape, device=enc_obs.device)

        seq_x[:, :, self.obs_idxes] = enc_obs
        if obs_first:
            seq_x[1:, :, self.act_idxes] = enc_act
        else:
            seq_x[:, :, self.act_idxes] = enc_act

        return seq_x

    def infer(self, input: Tensor, h0: Tensor | None = None):
        deter, hx = self.infer_rnn(input, h0)
        stoch_logits = self._stoch_fc(deter)
        state_dists = StateDist(
            deter=deter,
            logits=stoch_logits,
            num_tokens=self.stoch_tokens,
            vocab_size=self.stoch_vocab_size,
        )
        return state_dists, hx

    def imagine(self, state: State, enc_act: Tensor):
        x = torch.cat((state.stoch, enc_act), -1)
        next_deter = self.deter_cell(x, state.deter)
        next_stoch_logits = self._stoch_fc(next_deter)
        return StateDist(
            deter=next_deter,
            logits=next_stoch_logits,
            num_tokens=self.stoch_tokens,
            vocab_size=self.stoch_vocab_size,
        )

    def decode_obs(self, state: State):
        logits: Tensor = self._obs_dec(state.stoch)
        logits = logits.reshape(
            *logits.shape[:-1],
            self.spec.obs.num_tokens,
            self.spec.obs.vocab_size,
        )
        return D.Categorical(logits=logits)


class Trainer(nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=3e-4, eps=1e-5)

    def opt_step(self, batch: data.SliceBatch):
        losses: dict[str, Tensor] = {}
        coef = {"vq_commit": 0.25, "repr": 0.25}

        enc_obs = self.wm.obs_emb(batch.obs)
        enc_act = self.wm.act_emb(batch.act)
        infer_x = self.wm.pack_infer(enc_obs, enc_act, encoded=True)

        infer_dists, _ = self.wm.infer(infer_x)
        state = infer_dists.rsample()

        obs_pred_dists = self.wm.decode_obs(state)
        losses["obs"] = -obs_pred_dists.log_prob(batch.obs).mean()

        pred_dist = over_seq(self.wm.imagine)(state[:-1], enc_act)
        prior, post = infer_dists[1:], pred_dist
        losses["pred"] = D.kl_divergence(prior.detach(), post).mean()
        losses["repr"] = D.kl_divergence(prior, post.detach()).mean()

        loss: Tensor = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics = {f"{k}_loss": v for k, v in losses.items()}
        metrics["loss"] = loss
        return metrics

    def preview(self, batch: data.SliceBatch, tok: Tokenizer):
        seq_len, batch_size = batch.act.shape[:2]
        prefix_len = seq_len // 2

        enc_obs = self.wm.obs_emb(batch.obs)
        enc_act = self.wm.act_emb(batch.act)

        infer_x = self.wm.pack_infer(
            obs=enc_obs[: prefix_len + 1],
            act=enc_act[:prefix_len],
            encoded=True,
        )
        infer_dists, _ = self.wm.infer(infer_x)
        states = infer_dists.sample()

        states = [*states]
        for t in range(prefix_len, seq_len):
            next_state_dist = self.wm.imagine(states[-1], enc_act[t])
            state = next_state_dist.sample()
            states.append(state)

        states = torch.stack(states)
        pred_obs = over_seq(self.wm.decode_obs)(states).mode

        orig = over_seq(tok.obs.decode)(batch.obs)
        recon = over_seq(tok.obs.decode)(pred_obs)

        frames = []
        for t in range(seq_len):
            grid = []
            for i in range(batch_size):
                orig_ = data.to_pil(orig[t, i])
                recon_ = data.to_pil(recon[t, i])
                grid.append([orig_, recon_])
            grid = make_grid(grid)
            grid = np.asarray(grid.convert("RGB"))
            if t >= prefix_len:
                # Make predicted frames have a red tint
                grid = grid.copy()
                grid[..., 0] = np.ceil(0.75 * grid[..., 0] + 0.25 * 255)
            frames.append(grid)

        return ImageSequenceClip(frames, fps=1.0)
