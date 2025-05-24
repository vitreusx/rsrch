from typing import Any


def to_tree(leaves: dict[str, Any], sep: str = "."):
    res = {}
    for k, v in leaves.items():
        parts = k.split(sep)
        cur = res
        for p in parts[:-1]:
            if p not in cur:
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return res


def leaves(tree: dict[str, Any], sep: str = "."):
    res = {}

    def walk(cur, path: str = ""):
        if isinstance(cur, dict):
            for k, v in cur.items():
                subpath = k if len(path) == 0 else path + sep + k
                walk(v, subpath)
        else:
            res[path] = cur

    walk(tree)
    return res
