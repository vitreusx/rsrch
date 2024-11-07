import json
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


def pretrain_loader():
    results_dir = Path("../results/pretrain")
    scalars = TBScalars(".cache/pretrain")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"wm_ratio": int, "rl_ratio": int, "rl_freq": float, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        scores = df[df["tag"] == "val/mean_ep_ret"]["value"]
        test_r["score"] = scores.iloc[-1]
        res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    tags = []
    for _, row in res_df.iterrows():
        tags.append(f"{row['wm_ratio']}/{row['rl_ratio']}")
    res_df["tag"] = tags

    return res_df, scalars


def sanity_check_loader():
    results_dir = Path("../results/sanity_check")
    scalars = TBScalars(".cache/sanity_check")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        env, seed = test.name.split("-")
        seed = int(seed.removeprefix("seed="))
        res_df.append({"path": test, "env": env, "seed": seed})
        scalars.read(test)
    res_df = pd.DataFrame.from_records(res_df)
    res_df

    return res_df, scalars


def reference_loader():
    with open("ref_scores/baselines.json", "rb") as f:
        baselines = json.load(f)

    records = []
    for task in baselines:
        records.append(
            {
                "task": task.removeprefix("atari_"),
                **{
                    k: baselines[task].get(k)
                    for k in ("random", "human_gamer", "human_record")
                },
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def dreamerv2_loader():
    with open("ref_scores/atari-dreamerv2.json", "rb") as f:
        scores = json.load(f)

    records = []
    for run in scores:
        name = run["task"].removeprefix("atari_")
        name = "".join(w.capitalize() for w in name.split("_"))
        for x, y in zip(run["xs"], run["ys"]):
            records.append(
                {
                    "task": name,
                    "seed": int(run["seed"]),
                    "time": x,
                    "score": y,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df
