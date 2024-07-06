import torch


class Policy:
    compute_dtype: torch.dtype | None


policy = Policy()


def set_policy(compute_dtype=None):
    policy.compute_dtype = compute_dtype


def autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=policy.compute_dtype,
        enabled=policy.compute_dtype is not None,
    )
