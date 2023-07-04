import torch
from torch import Tensor


def fix_log_prob_(logits: Tensor, max_abs_value=100.0):
    """Fixes (in-place) a tensor of logits by (1) replacing NaNs with mean value, (2) clipping the values from below. In some cases, sampling procedure for some distributions may yield a sample for which log probability is either excessively low or outright NaN. This prevents any errors arising from such situations."""

    mask = torch.isnan(logits)
    replace = torch.nanmean(logits).detach()
    logits.masked_fill_(mask, replace)
    logits.clamp_min_(-max_abs_value)
