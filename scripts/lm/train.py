from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from platformdirs import user_cache_dir
from ruamel.yaml import YAML
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from datasets import DatasetDict, load_dataset
from rsrch.exp import Experiment, boards
from rsrch.lang.data import BucketBatchSampler, split_into_buckets
from rsrch.models.transformer import Decoder
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.utils import cron, ddp, repro
from rsrch.utils.cast import cast
from rsrch.utils.path import sanitize

# isort: off
from config import Config, TimeDelta
# isort: on


class Item(TypedDict):
    """Item type of the base dataset."""

    text: str


def compute_seq_lengths(data: Iterable[Item]):
    result = []
    for item in tqdm(data, desc="Computing sentence lengths"):
        seq_len = len(item["text"])
        result.append(seq_len)
    return np.array(result)


class DatasetForLM(Sequence[Item]):
    """A dataset wrapper for language modelling.

    Adds BOS and EOS tokens to the text, and adds a restriction on the maximum
    size of the sequence - in the case of long sequences, a slice thereof is
    taken."""

    def __init__(
        self,
        corpus: Sequence[Item],
        max_seq_len: int,
        bos_token: str,
        eos_token: str,
        seed: int = 0,
    ):
        self.corpus = corpus
        self.max_seq_len = max_seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.gen = np.random.default_rng(seed=seed)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index: int):
        text = self.corpus[index]["text"]
        if len(text) <= self.max_seq_len - 2:
            return self.bos_token + text + self.eos_token
        else:
            # Get slice from (bos + text + eos) without constructing the
            # (possibly long) string explicitly
            begin = int(self.gen.choice(len(text) + 2 - self.max_seq_len))
            end = begin + self.max_seq_len
            if begin == 0:
                return self.bos_token + text[: end - 1]
            elif end == len(text):
                return text[begin - 1 :] + self.eos_token
            else:
                return text[begin - 1 : end - 1]


class Trainer:
    project = "lm"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        self.setup_base()
        self.setup_tokenizer()
        self.setup_data()
        self.setup_model()

        def get_flag(delta: TimeDelta | None):
            if delta is None:
                return cron.Never()
            else:
                return cron.Every(
                    step_fn=lambda: getattr(self, delta.of),
                    period=delta.n,
                )

        self.should_val = get_flag(self.cfg.val_every)

        self.step = 0
        while True:
            if self.should_val:
                self.val_epoch()
            self.train_step()
            self.step += 1

    def setup_base(self):
        self.ddp = ddp.auto_detect()
        repro.seed_all(self.cfg.seed)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        self.step, self.epoch = 0, 0
        if self.ddp.is_master:
            self.exp = Experiment(
                project=self.project,
                create_commit=self.cfg.create_exp_commit,
            )
            self.exp.add_board(boards.Tensorboard(self.exp.dir / "board", launch=True))

            self.exp.register_step("step", lambda: self.step)
            self.exp.register_step("epoch", lambda: self.epoch)

    def setup_tokenizer(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path,
            extra_special_tokens={
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
            },
            add_bos_token=False,
            add_eos_token=False,
        )

    def setup_data(self):
        data = load_dataset(self.cfg.dataset_path, self.cfg.dataset_name)
        assert isinstance(data, DatasetDict)
        assert "train" in data and "validation" in data

        ident = f"datasets/{self.cfg.dataset_path}"
        if self.cfg.dataset_name is not None:
            ident = ident + f"/{self.cfg.dataset_name}"
        ident = sanitize(ident, repl="--")
        cache_dir = Path(user_cache_dir("rsrch")) / ident

        lengths = {}
        datasets = {}
        data_loaders = {}

        for split in ("train", "validation"):
            cache_path = cache_dir / f"seq_lengths_{split}.npy"
            if not cache_path.exists():
                lengths[split] = compute_seq_lengths(data[split])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    np.save(f, lengths[split])
            else:
                with open(cache_path, "rb") as f:
                    lengths[split] = np.load(f)

            datasets[split] = DatasetForLM(
                data[split],
                max_seq_len=self.cfg.max_seq_len,
                bos_token=self.tokenizer.bos_token,
                eos_token=self.tokenizer.eos_token,
                seed=0,
            )
            lengths[split] = lengths[split].clip(max=self.cfg.max_seq_len)

            buckets = split_into_buckets(
                lengths=lengths[split],
                min_seq_len=self.cfg.min_seq_len,
                min_bucket_size=self.cfg.min_bucket_size,
                min_bucket_token_count=self.cfg.min_bucket_char_count,
            )
            batch_sampler = BucketBatchSampler(
                lengths=lengths[split],
                buckets=buckets,
                tokens_per_batch=self.cfg.chars_per_batch,
                shuffle=(split == "train"),
                drop_last=(split == "train"),
                seed=self.epoch,
            )
            data_loaders[split] = DataLoader(
                dataset=datasets[split],
                batch_sampler=self.ddp.wrap_sampler(
                    batch_sampler,
                    set_epoch=batch_sampler.set_epoch,
                    drop_last=(split == "train"),
                ),
            )

        self.train_loader = data_loaders["train"]
        self.train_iter = self.get_train_iter()
        self.val_loader = data_loaders["validation"]

        # Get a sample batch for showing text completion results
        val_len = lengths["validation"]
        sample_idxes = np.nonzero(val_len >= self.cfg.sample_min_seq_len)[0]
        val_data = datasets["validation"]

        val_sample = []
        for idx in sample_idxes[: self.cfg.sample_size].tolist():
            seq: str = val_data[idx]
            words = seq.split(" ")
            min_words, max_words = int(0.25 * len(words)), int(0.75 * len(words))
            word_count = int(np.random.randint(min_words, max_words + 1))
            seq = " ".join(words[:word_count])
            val_sample.append(seq)

        self.val_sample = data_loaders["validation"].collate_fn(val_sample)

    def get_train_iter(self):
        self.epoch = 0
        while True:
            self.ddp.set_epoch(self.train_loader.batch_sampler, self.epoch)
            yield from self.train_loader
            self.epoch += 1

    def setup_model(self):
        key_dim = value_dim = self.cfg.model_dim // self.cfg.num_heads
        self.model = Decoder(
            vocab_size=len(self.tokenizer),
            model_dim=self.cfg.model_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_blocks=self.cfg.num_blocks,
            num_heads=self.cfg.num_heads,
            padding_idx=self.tokenizer.pad_token_id,
        )

        self.model = self.ddp.wrap_model(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)

    def train_step(self):
        batch = next(self.train_iter)

        output = self.tokenizer(
            batch, return_tensors="pt", padding=True, padding_side="right"
        )
        input_ids: Tensor = output["input_ids"]
        input_ids = input_ids.to(self.ddp.device).T  # [N, L] -> [L, N]
        attn_mask: Tensor = output["attention_mask"]
        attn_mask: Tensor = attn_mask.to(device=self.ddp.device, dtype=torch.bool)

        with self.autocast():
            logits = self.model(input_ids, attn_mask=attn_mask)
            per_token_loss = F.cross_entropy(
                logits[:-1].reshape(-1, logits.shape[-1]),
                input_ids[1:].flatten(),
                reduction="none",
            )
            token_weight = attn_mask[:, :-1].flatten().type_as(per_token_loss)
            loss = (per_token_loss * token_weight).mean()

        self.opt.step(loss)

        if self.ddp.is_master:
            self.exp.add_scalar("train/loss", loss, step=self.step)

    def val_epoch(self):
        if not self.ddp.is_master:
            return

        sample_text = self.val_sample
        completions = self.sample(sample_text)

        result_text = ["# Completions"]
        for idx in range(len(sample_text)):
            result_text.append(
                f"""
## Sample #{idx:03d}

Prompt:
```
{sample_text[idx]}
```

Completion:
```
(...) {completions[idx]}
```"""
            )

        result_text = "\n".join(result_text)

        unit = self.cfg.val_every.of
        time_val = self.should_val.step_fn()
        time = f"{unit}={time_val:07d}"

        dest = self.exp.dir / "samples" / f"{time}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            f.write(result_text)

        self.exp.info(f"Completions saved to {dest}")

    def sample(self, src_text: list[str]):
        batch_size = len(src_text)
        out_indices = np.arange(batch_size)
        outputs = [None for _ in range(batch_size)]

        output = self.tokenizer(
            src_text,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )

        input_ids = output["input_ids"].T.to(device=self.ddp.device)
        attn_mask = output["attention_mask"].to(
            device=self.ddp.device, dtype=torch.bool
        )
        offsets = [input_ids.shape[0] for _ in range(batch_size)]

        self.model.eval()

        eos_token_id = self.tokenizer.eos_token_id
        for _ in range(self.cfg.max_completion_tokens):
            is_eos: Tensor = input_ids[-1] == eos_token_id
            if is_eos.any():
                indices = torch.where(is_eos)[0].numpy(force=True)
                for idx in indices:
                    out_idx = out_indices[idx]
                    seq_tokens = input_ids[offsets[out_idx] :, idx]
                    outputs[out_idx] = seq_tokens.numpy(force=True).tolist()

                if is_eos.all():
                    break

                mask = ~is_eos
                input_ids, attn_mask = input_ids[:, mask], attn_mask[mask]
                out_indices = out_indices[mask.numpy(force=True)]

            last_hidden = self.model.encode(input=input_ids, attn_mask=attn_mask)
            logits = self.model.proj(last_hidden[-1:]).squeeze(0)
            next_token_ids = logits.argmax(-1)

            input_ids = torch.cat((input_ids, next_token_ids[None]), dim=0)
            next_tokens_mask = torch.ones(
                (attn_mask.shape[0], 1), dtype=attn_mask.dtype, device=attn_mask.device
            )
            attn_mask = torch.cat((attn_mask, next_tokens_mask), dim=1)

        self.model.train()

        for idx in range(len(out_indices)):
            out_idx = out_indices[idx]
            seq_tokens = input_ids[offsets[out_idx] :, idx]
            outputs[out_idx] = seq_tokens.numpy(force=True).tolist()

        return self.tokenizer.batch_decode(outputs)

    def autocast(self):
        return torch.autocast(
            self.ddp.device.type,
            self.compute_dtype,
            enabled=self.compute_dtype != torch.float32,
        )


def main():
    """Main function."""
    yaml = YAML(typ="safe", pure=True)
    with open(Path(__file__).parent / "config.yml", "r") as f:
        cfg = cast(yaml.load(f), Config)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
