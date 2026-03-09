import math

import numpy as np


def split_into_buckets(
    lengths: np.ndarray,
    min_seq_len: int | None = None,
    max_seq_len: int | None = None,
    min_bucket_size: int | None = None,
    min_bucket_token_count: int | None = None,
    drop_last: bool = False,
):
    """Split sequences into buckets by lengths.

    :param lengths: A list of lengths of the sequences.
    :param min_seq_len: (Optional) Minimum sequence length to consider.
    :param max_seq_len: (Optional) Maximum sequence length to consider.
    :param min_bucket_size: (Optional) Minimum number of sequences in a single bucket.
    :param min_bucket_token_count: (Optional) Minimum number of tokens in a
        single bucket.
    :param drop_last: Whether to drop the last bucket, if it doesn't meet the
        bucket requirements. If false, the last and penultimate buckets are
        merged.
    """

    if min_bucket_size is None and min_bucket_token_count is None:
        raise ValueError(
            "One of 'min_bucket_size', 'min_bucket_tok_count' must be provided"
        )

    indices = np.argsort(lengths)
    values = lengths[indices]

    i_min, i_max = 0, len(lengths)
    if min_seq_len is not None:
        # Find least idx s.t. values[idx] >= min_seq_len
        i_min = np.searchsorted(values, min_seq_len)
    if max_seq_len is not None:
        # Find least idx s.t. values[idx] > max_seq_len
        i_max = np.searchsorted(values, max_seq_len, side="right")

    cumul_tok_counts = np.cumsum(values)

    buckets = []
    begin = i_min
    while begin < i_max:
        end = begin + 1

        if min_bucket_size is not None:
            end_by_size = begin + min_bucket_size
            end = max(end, end_by_size)

        if min_bucket_token_count is not None:
            # tok_count for bucket is cumul[last - 1] - cumul[begin - 1]
            # so: search for least idx s.t.
            # > cumul[idx] >= cumul[begin - 1] + min_bucket_tok_count
            # and take last := idx + 1
            cur_tok_count = cumul_tok_counts[begin - 1] if begin > 0 else 0
            max_tok_count = cur_tok_count + min_bucket_token_count
            end_by_tok_count = np.searchsorted(cumul_tok_counts, max_tok_count) + 1
            end = max(end, end_by_tok_count)

        if end <= i_max:
            buckets.append((begin, end))
        elif not drop_last:
            # Expand the last bucket
            if len(buckets) > 0:
                prev_begin, _ = buckets[-1]
                buckets[-1] = (prev_begin, i_max)
            else:
                buckets.append((begin, i_max))

        begin = end

    buckets = [indices[start:end] for start, end in buckets]
    return buckets


class BucketBatchSampler:
    """Bucketed batch sampler."""

    def __init__(
        self,
        lengths: np.ndarray,
        buckets: list[np.ndarray],
        batch_size: int | None = None,
        tokens_per_batch: int | None = None,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.tokens_per_batch = tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._seed = seed
        self.set_epoch(0)

        self.buckets = []
        if batch_size is None:
            for bucket in buckets:
                avg_len = lengths[bucket].mean()
                batch_size = math.ceil(tokens_per_batch / avg_len)
                self.buckets.append((bucket, batch_size))
        else:
            for bucket in buckets:
                self.buckets.append((bucket, batch_size))

    def set_epoch(self, epoch: int):
        self.seed = self._seed + epoch

    def __iter__(self):
        if self.shuffle:
            gen = np.random.default_rng(seed=self.seed)
            buckets, batches = [], []
            for idx, (bucket, batch_size) in enumerate(self.buckets):
                buckets.append(gen.permuted(bucket))
                for offset in range(0, len(bucket), batch_size):
                    if self.drop_last and offset + batch_size > len(bucket):
                        break
                    end = min(offset + batch_size, len(bucket))
                    start = max(end - batch_size, 0)
                    batches.append((idx, slice(start, end)))

            for idx in gen.permutation(len(batches)):
                bucket_idx, item_idx = batches[idx]
                yield buckets[bucket_idx][item_idx].tolist()

        else:
            for bucket, batch_size in self.buckets:
                for offset in range(0, len(bucket), batch_size):
                    if self.drop_last and offset + batch_size > len(bucket):
                        break
                    end = min(offset + batch_size, len(bucket))
                    start = max(end - batch_size, 0)
                    yield bucket[start:end].tolist()

    def __len__(self):
        count = 0
        for bucket_idx in range(len(self.buckets)):
            bucket, batch_size = self.buckets[bucket_idx]
            if self.drop_last:
                count += len(bucket) // batch_size
            else:
                count += (len(bucket) + batch_size - 1) // batch_size
        return count
