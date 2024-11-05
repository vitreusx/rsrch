from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from utils import TBScalars


def data_aug_loader():
    results_dir = Path("../results/data_aug_2")
    scalars = TBScalars(".cache/data_aug_2")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        scores = df[df["tag"] == "val/mean_ep_ret"]["value"]
        test_r["score"] = scores.iloc[-1]
        res_df.append({"path": test, **test_r})

    res_df = pd.DataFrame.from_records(res_df)
    return res_df, scalars


def baseline_loader():
    results_dir = Path("../results/baseline")
    scalars = TBScalars(".cache/baseline")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        scores = df[df["tag"] == "val/mean_ep_ret"]["value"]
        test_r["score"] = scores.iloc[-1]
        res_df.append({"path": test, **test_r})

    res_df = pd.DataFrame.from_records(res_df)
    return res_df, scalars
