from tqdm.auto import tqdm


class ProgressBar:
    def __init__(self, desc: str, total: int):
        self._pbar = tqdm(total=total, desc=desc, dynamic_ncols=True)

    def step(self, n=1):
        self._pbar.update(n)
