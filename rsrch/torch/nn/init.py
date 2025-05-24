import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch import Tensor, nn


def _sample_pivots_truncnorm(output: np.ndarray, seed: np.random.Generator):
    """Sample pivots in such a way, that for each feature the position of the pivot is in the quantile `q`, where `q` is sampled from truncated-normal distribution."""

    batch_size, out_features = output.shape
    sorted_out = np.sort(output, axis=0)

    loc, scale, min_v, max_v = 0.5, 0.2, 0.1, 0.9
    a, b = (min_v - loc) / scale, (max_v - loc) / scale
    pivots = truncnorm.rvs(a, b, loc, scale, size=out_features, random_state=seed)
    idxes = (pivots * batch_size).astype(np.int32)

    return sorted_out[idxes, np.arange(out_features)]


def _sample_pivots_unif(output: np.ndarray, seed: np.random.Generator):
    """Sample pivots at random."""

    batch_size, out_features = output.shape
    idxes = seed.integers(batch_size, size=out_features)
    return output[idxes, np.arange(out_features)]


def _linear_relu_init(input: np.ndarray, out_features: int, seed=None):
    """Prepare weight and bias matrix for a linear/dense layer in such a way,
    that the output is (approximately) zero-centered for a given sample input,
    and the features are random and approximately of scale one."""

    g = np.random.default_rng(seed=seed)

    in_features = input.shape[1]
    weight_t = g.normal(size=(in_features, out_features))
    output = np.matmul(input, weight_t)  # (N, d_out)

    pivots = _sample_pivots_unif(output, seed=g)
    bias = -pivots

    output = output + bias
    scale = np.std(output, axis=0)  # (d_out)
    weight_t, bias = weight_t / scale, bias / scale

    weight = weight_t.T
    return weight, bias


@torch.inference_mode()
def linear_relu_init_(layer: nn.Linear, input: Tensor):
    """Initialize weights for `nn.Linear` layer."""

    if layer.bias is None:
        raise RuntimeError("Unsupported nn.Linear configuration.")

    sample_nd = input.numpy()
    weight, bias = _linear_relu_init(sample_nd, layer.out_features)

    layer.weight.data.copy_(torch.as_tensor(weight))
    layer.bias.data.copy_(torch.as_tensor(bias))


@torch.inference_mode()
def conv2d_relu_init_(layer: nn.Conv2d, input: Tensor):
    """Initialize weights for `nn.Conv2d` layer."""

    if layer.groups != 1 or layer.bias is None:
        raise RuntimeError("Unsupported nn.Conv2d configuration.")

    input_col = F.unfold(
        input,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )
    input_col = input_col.moveaxis(2, 1).flatten(0, 1)
    weight, bias = _linear_relu_init(input_col.numpy(), layer.out_channels)
    weight = torch.as_tensor(weight).reshape_as(layer.weight)

    layer.weight.data.copy_(weight)
    layer.bias.data.copy_(torch.as_tensor(bias))
