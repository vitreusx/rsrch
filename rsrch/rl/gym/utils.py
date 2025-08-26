from typing import Iterable


def episodes(steps: Iterable[tuple[int, tuple[dict, bool]]]):
    eps: dict[int, list[dict]] = {}
    for env_idx, (step, final) in steps:
        if env_idx not in eps:
            eps[env_idx] = []
        eps[env_idx].append(step)
        if final:
            yield env_idx, eps[env_idx]
            del eps[env_idx]
