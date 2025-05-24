from torch import Tensor

_KL_DIVS = {}


def kl_divergence(p, q) -> Tensor:
    if (type(p), type(q)) not in _KL_DIVS:
        dist, func = None, None
        for p_idx, p_super in enumerate(type(p).__mro__):
            for q_idx, q_super in enumerate(type(q).__mro__):
                if (p_super, q_super) in _KL_DIVS:
                    cur_dist = p_idx + q_idx
                    if dist is None or cur_dist < dist:
                        dist = cur_dist
                        func = _KL_DIVS[p_super, q_super]

        if func is not None:
            _KL_DIVS[type(p), type(q)] = func
        else:
            msg = f"KL divergence for ({type(p)}, {type(q)}) not registered"
            raise ValueError(msg)

    return _KL_DIVS[type(p), type(q)](p, q)


def register_kl(ptype, qtype):
    def _decorator(_func):
        _KL_DIVS[ptype, qtype] = _func
        return _func

    return _decorator
