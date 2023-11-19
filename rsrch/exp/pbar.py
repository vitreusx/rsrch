from functools import partial

from tqdm.auto import tqdm

ProgressBar = partial(tqdm, dynamic_ncols=True)
