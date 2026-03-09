def flatten(x, path=None):
    if isinstance(x, dict):
        flat = {}
        for k, v in x.items():
            v_path = f"{k}" if path is None else f"{path}.{k}"
            flat.update(flatten(v, v_path))
        return flat
    elif isinstance(x, list):
        flat = {}
        for i, v in enumerate(x):
            v_path = f"{i}" if path is None else f"{path}.{i}"
            flat.update(flatten(v, v_path))
        return flat
    else:
        return {path: x}
