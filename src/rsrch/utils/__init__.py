import importlib.util


def is_torch_available():
    return importlib.util.find_spec("torch") is not None


def is_jax_available():
    return importlib.util.find_spec("jax") is not None


def is_envpool_available():
    return importlib.util.find_spec("envpool") is not None
