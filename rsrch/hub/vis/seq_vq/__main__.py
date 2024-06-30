import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from moviepy.editor import *
from PIL import Image
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import exp, spaces
from rsrch.nn import vq
from rsrch.nn.utils import over_seq, pass_gradient, safe_mode
from rsrch.rl import data
from rsrch.utils import cron, repro
from rsrch.utils.data import DataLoader, InfiniteSampler
from rsrch.utils.preview import make_grid

from . import config, nets


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()

        obs_space = spaces.torch.Image((1, 64, 64), torch.float32)
        act_space = spaces.torch.Discrete(18)

        self.obs_norm = nets.Normalize()
        self.obs_enc = nets.Encoder_v1(obs_space)
        with safe_mode(self):
            obs = obs_space.sample([1])
            self.obs_size = self.obs_enc(obs).shape[1]

        self.act_enc = nn.Embedding(act_space.n, 128)
        with safe_mode(self):
            act = act_space.sample([1])
            self.act_size = self.act_enc(act).shape[1]

        self.seq_hidden, self.seq_layers = 1024, 2

        self._infer_h0 = nn.Parameter(torch.randn(self.seq_layers, 1, self.seq_hidden))

        self.infer_rnn = nn.GRU(
            input_size=self.act_size + self.obs_size,
            hidden_size=self.seq_hidden,
            num_layers=self.seq_layers,
        )

        self.z_enc_rnn = nn.GRU(
            input_size=self.act_size + self.obs_size,
            hidden_size=self.seq_hidden,
            num_layers=self.seq_layers,
        )

        self.z_tokens, self.z_vocab, self.z_embed = 8, 128, 256
        self.z_vq = vq.VSQLayer(
            num_tokens=self.z_tokens,
            vocab_size=self.z_vocab,
            embed_dim=self.z_embed,
        )

        self.z_size = self.z_tokens * self.z_embed
        self.z_proj = nn.Sequential(
            nn.Linear(self.seq_hidden, self.z_size),
            nets.Reshape(-1, (self.z_tokens, self.z_embed)),
        )

        self.z_dec_rnn = nn.GRU(
            input_size=self.act_size + self.z_size,
            hidden_size=self.seq_hidden,
            num_layers=self.seq_layers,
        )

        self.obs_dec = nets.Decoder_v1(self.seq_hidden, obs_space)

    def infer_h0(self, batch_size: int):
        h0 = self._infer_h0.repeat(1, batch_size, 1)
        return h0.contiguous()


class Dataset:
    def __init__(
        self,
        buf_data: data.BufferData,
        slice_len: int,
        batch_size: int,
        ongoing: bool = True,
        subseq_len: int | tuple[int, int] | None = None,
        min_term_prob: float = 0.0,
    ):
        self.eps = data.EpisodeView(data.Buffer(buf_data))
        self.slice_len = slice_len
        self.batch_size = batch_size
        self.ongoing = ongoing
        self.min_term_prob = min_term_prob

        if subseq_len is None:
            self.minlen, self.maxlen = 1, None
        elif isinstance(subseq_len, tuple):
            self.minlen, self.maxlen = subseq_len
        else:
            self.minlen, self.maxlen = subseq_len, subseq_len
        self.minlen = max(self.minlen, self.slice_len)

    def __iter__(self):
        cur_eps = {}
        free_idx = 0

        while True:
            batch = {}

            self.eps.update()
            min_id, max_id = self.eps.ids.start, self.eps.ids.stop

            # Remove subsequences which have ran out
            for idx in [*cur_eps]:
                ep, start, stop = cur_eps[idx]
                if start >= stop:
                    del cur_eps[idx]

            # Fill the batch quota with new subsequences
            while len(cur_eps) < self.batch_size:
                ep_id = np.random.randint(min_id, max_id)
                ep = self.eps[ep_id]
                if not self.ongoing and not ep.term:
                    continue

                total = len(ep.obs)
                if total < self.minlen:
                    continue

                # Determine random subseq len divisible by the slice_len
                subseq_len = np.random.randint(self.minlen, self.maxlen + 1)
                subseq_len = min(subseq_len, total)
                subseq_len = self.slice_len * (subseq_len // self.slice_len)

                # If necessary, oversample the final subsequence to make the
                # terminality probability sufficiently high
                num_subseq = total - subseq_len + 1
                batches_per_subseq = subseq_len / self.slice_len
                term_prob = (1.0 / num_subseq) / batches_per_subseq

                if term_prob < self.min_term_prob:
                    start = np.random.randint(total - subseq_len)
                    if np.random.rand() < self.min_term_prob:
                        start = total - subseq_len
                else:
                    start = np.random.randint(total - subseq_len + 1)

                stop = start + subseq_len
                cur_eps[free_idx] = [ep, start, stop]
                free_idx += 1

            # Yield the subsequence slices
            for idx in cur_eps:
                ep, start, stop = cur_eps[idx]
                batch[idx] = {"seq": ep[start:stop], "is_first": start == 0}
                cur_eps[idx] = [ep, start + self.slice_len, stop]

            yield batch


class SeqVQ(nn.Module):
    def run(self):
        cfg = config.load(Path(__file__).parent / "config.yml")
        self.cfg = config.parse(cfg, config.Config)

        repro.fix_seeds(seed=self.cfg.seed, deterministic=False)

        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype)

        self.opt_step = 0

        env_id = self.cfg.samples.stem.split("__")[0]

        self.exp = exp.Experiment(
            project="seq_vq",
            run=f"{env_id}__{exp.timestamp()}",
            config=cfg,
        )
        self.exp.boards += [exp.board.Tensorboard(self.exp.dir / "board")]
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.wm = WorldModel().to(self.device)
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=3e-4, eps=1e-5)
        self.scaler = getattr(torch, self.device.type).amp.GradScaler()

        with open(self.cfg.samples, "rb") as f:
            buf_data = pickle.load(f)["samples"]

        self.train_ds = Dataset(buf_data, **vars(self.cfg.dataset))
        self.train_iter = iter(self.train_ds)

        def train_step():
            batch = next(self.train_iter)
            batch = [item["seq"] for item in batch.values()]
            batch: data.SliceBatch = data.default_collate_fn(batch)
            batch = batch.to(self.device)
            batch.obs = over_seq(tv_F.resize)(batch.obs / 255.0, size=(64, 64))

            batch_size = self.train_ds.batch_size

            losses = {}
            coef = {}

            with self.autocast():
                norm_obs = over_seq(self.wm.obs_norm)(batch.obs)
                enc_obs = over_seq(self.wm.obs_enc)(norm_obs)

                norm_act = batch.act
                enc_act = over_seq(self.wm.act_enc)(norm_act)

                h0 = self.wm.infer_h0(batch_size)

                prefix_len = self.cfg.prefix_len
                act = torch.cat((torch.zeros_like(enc_act[:1]), enc_act), 0)
                seq_x = torch.cat((act, enc_obs), -1)

                _, h0 = self.wm.infer_rnn(seq_x[:prefix_len], h0.contiguous())

                out, _ = self.wm.z_enc_rnn(seq_x[prefix_len:], h0)

                out = self.wm.z_proj(out)
                z, z_idx = self.wm.z_vq(out)
                z_vq_embed = F.mse_loss(z, out.detach())
                z_vq_commit = F.mse_loss(z.detach(), out)
                losses["vq"] = 0.8 * z_vq_embed + 0.2 * z_vq_commit
                z = pass_gradient(z, out).flatten(-2)

                seq_x = torch.cat((enc_act[prefix_len - 1 :], z), -1)
                out, _ = self.wm.z_dec_rnn(seq_x, h0)

                recon = over_seq(self.wm.obs_dec)(out)
                losses["recon"] = F.mse_loss(recon, norm_obs[prefix_len:])

                loss = sum(coef[k] * v if k in coef else v for k, v in losses.items())

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.wm.parameters(), 10.0)
            self.scaler.step(self.opt)
            self.scaler.update()

            self.exp.add_scalar("train/loss", loss)
            for k, v in losses.items():
                self.exp.add_scalar(f"train/{k}_loss", v)

        self.test_ds = data.SliceView(data.Buffer(buf_data), 128)
        self.test_loader = DataLoader(
            dataset=self.test_ds,
            batch_size=8,
            sampler=InfiniteSampler(self.test_ds),
            collate_fn=data.default_collate_fn,
            drop_last=True,
            worker_init_fn=repro.worker_init_fn,
        )
        self.test_iter = iter(self.test_loader)

        should_test = cron.Every(lambda: self.opt_step, self.cfg.test_every)

        @safe_mode(self)
        def test_epoch():
            batch = next(self.test_iter)
            batch = batch.to(self.device)
            batch.obs = over_seq(tv_F.resize)(batch.obs / 255.0, size=(64, 64))

            with self.autocast():
                norm_obs = over_seq(self.wm.obs_norm)(batch.obs)
                enc_obs = over_seq(self.wm.obs_enc)(norm_obs)

                norm_act = batch.act
                enc_act = over_seq(self.wm.act_enc)(norm_act)

                batch_size = batch.obs.shape[1]
                h0 = self.wm.infer_h0(batch_size)

                prefix_len = self.cfg.prefix_len
                act = torch.cat((torch.zeros_like(enc_act[:1]), enc_act), 0)
                seq_x = torch.cat((act, enc_obs), -1)

                _, h0 = self.wm.infer_rnn(seq_x[:prefix_len], h0.contiguous())

                out, _ = self.wm.z_enc_rnn(seq_x[prefix_len:], h0)

                out = self.wm.z_proj(out)
                z, z_idx = self.wm.z_vq(out)
                z = z.flatten(-2)

                seq_x = torch.cat((enc_act[prefix_len - 1 :], z), -1)
                out, _ = self.wm.z_dec_rnn(seq_x, h0)

                recon = over_seq(self.wm.obs_dec)(out)

                orig = batch.obs[prefix_len:]
                recon = over_seq(self.wm.obs_norm.inv)(recon)

            def to_pil(x: Tensor):
                x = (255 * x.clamp(0.0, 1.0)).to(torch.uint8)
                if x.shape[0] == 1:
                    x = x.squeeze(0)
                return Image.fromarray(x.cpu().numpy())

            frames = []
            seq_len, batch_size = orig.shape[:2]
            for t in range(seq_len):
                grid = []
                for i in range(batch_size):
                    orig_ = to_pil(orig[t, i])
                    recon_ = to_pil(recon[t, i])
                    grid.append([orig_, recon_])
                grid = make_grid(grid)
                frames.append(np.asarray(grid))

            vid = ImageSequenceClip(frames, fps=30.0)
            self.exp.add_video("test/samples", vid)

        self.pbar = tqdm(desc="SeqVQ")
        while True:
            if should_test:
                test_epoch()

            train_step()

            self.opt_step += 1
            self.pbar.update()

    def autocast(self):
        return torch.autocast(self.device.type, self.dtype)


if __name__ == "__main__":
    SeqVQ().run()
