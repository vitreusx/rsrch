import json
from pathlib import Path

import pandas as pd
from parse import *
from tqdm.auto import tqdm
from utils import TBScalars


def data_aug_v2_loader():
    results_dir = Path("../results/data_aug/v2")
    scalars = TBScalars(".cache/data_aug/v2")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
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
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
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
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
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
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 6e6:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)
    res_df

    return res_df, scalars


def canonical_task_name(name: str):
    name = name.removeprefix("atari_")
    name = "".join(w.capitalize() for w in name.split("_"))
    if name == "JamesBond":
        name = "Jamesbond"
    return name


def reference_loader():
    with open("ref_scores/baselines.json", "rb") as f:
        baselines = json.load(f)

    records = []
    for task in baselines:
        records.append(
            {
                "task": canonical_task_name(task),
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
        for x, y in zip(run["xs"], run["ys"]):
            records.append(
                {
                    "task": canonical_task_name(run["task"]),
                    "seed": int(run["seed"]),
                    "time": x,
                    "score": y,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def adaptive_ratio_v1_0_loader():
    results_dir = Path("../results/adaptive_ratio/v1_0")
    scalars = TBScalars(".cache/adaptive_ratio/v1_0")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)
    res_df

    return res_df, scalars


def adaptive_ratio_wm_v1_0_loader():
    results_dir = Path("../results/adaptive_ratio/wm_v1_0")
    scalars = TBScalars(".cache/adaptive_ratio/wm_v1_0")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"rl_ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)
    res_df

    return res_df, scalars


def split_ratios_loader():
    results_dir = Path("../results/split_ratios")
    scalars = TBScalars(".cache/split_ratios")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"wm_ratio": int, "rl_ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    res_df = res_df[res_df["env"].isin(("Assault",))]

    return res_df, scalars


def ppo_sweep_loader():
    results_dir = Path("../results/ppo/sweep")
    scalars = TBScalars(".cache/ppo/sweep")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"cfg": str, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))

        cfg = test_r["cfg"]
        rl_ratio, num_epochs, num_mb = parse("k{}_e{}_mb{}", cfg)
        del test_r["cfg"]
        test_r = {
            **test_r,
            "rl_ratio": rl_ratio,
            "num_epochs": num_epochs,
            "num_mb": num_mb,
        }

        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def data_aug_sweep_loader():
    results_dir = Path("../results/data_aug/sweep")
    scalars = TBScalars(".cache/data_aug/sweep")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"type": str, "ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})

    res_df = pd.DataFrame.from_records(res_df)
    return res_df, scalars


def sac_alpha_search_loader():
    results_dir = Path("../results/sac/alpha_search")
    scalars = TBScalars(".cache/sac/alpha_search")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int, "round": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def sac_ent_sched_exp_loader():
    results_dir = Path("../results/sac/ent_sched_exp")
    scalars = TBScalars(".cache/sac/ent_sched_exp")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def sac_ent_sched2_loader():
    results_dir = Path("../results/sac/ent_sched2")
    scalars = TBScalars(".cache/sac/ent_sched2")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def warm_start_loader():
    results_dir = Path("../results/warm_start")
    scalars = TBScalars(".cache/warm_start")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 6e6:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def final_prelim_loader():
    results_dir = Path("../results/final/prelim")
    scalars = TBScalars(".cache/final/prelim")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"cfg": str, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))

        cfg = test_r["cfg"]
        rl_ratio, num_epochs = (int(x) for x in cfg.split("x"))
        del test_r["cfg"]
        test_r = {
            **test_r,
            "rl_ratio": rl_ratio,
            "num_epochs": num_epochs,
        }

        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def warm_start_actor_loader():
    results_dir = Path("../results/warm_start_actor")
    scalars = TBScalars(".cache/warm_start_actor")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 6e6:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def final_benchmark_loader():
    results_dir = Path("../results/final/benchmark")
    scalars = TBScalars(".cache/final/benchmark")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def papers_loader():
    bbf_scores = pd.read_csv("ref_scores/bbf_scores.csv")
    dv3_v2_scores = pd.read_csv("ref_scores/dv3_v2_scores.csv")
    missing = {*dv3_v2_scores.columns} - {*bbf_scores.columns}
    scores = bbf_scores.merge(
        dv3_v2_scores[["Environment", *missing]],
        on="Environment",
    )
    dv2_scores = pd.read_csv("ref_scores/dv2_scores.csv")
    scores = scores.merge(
        dv2_scores,
        on="Environment",
    )

    bbf_stats = pd.read_csv("ref_scores/bbf_stats.csv")
    dv3_v2_stats = pd.read_csv("ref_scores/dv3_v2_stats.csv")
    missing = {*dv3_v2_stats.columns} - {*bbf_stats.columns}
    stats = bbf_stats.merge(
        dv3_v2_stats[["Statistic", *missing]],
        on="Statistic",
    )
    dv2_stats = pd.read_csv("ref_scores/dv2_stats.csv")
    stats = stats.merge(
        dv2_stats,
        on="Statistic",
    )

    return scores, stats


def final_dreamerv2_loader():
    results_dir = Path("../results/final/dreamerv2")
    scalars = TBScalars(".cache/final/dreamerv2")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars


def mpc_base_loader():
    results_dir = Path("../results/mpc/base")
    scalars = TBScalars(".cache/mpc/base")

    res_df = []
    for test in tqdm([*results_dir.iterdir()]):
        params = test.name.split("-")
        test_r = {}
        test_r["env"] = params[0]
        types = {"ratio": int, "seed": int}
        for (name, typ), value in zip(types.items(), params[1:]):
            test_r[name] = typ(value.removeprefix(f"{name}="))
        df = scalars.read(test)
        final_val = df[df["tag"] == "val/mean_ep_ret"].iloc[-1]
        if final_val["step"] >= 400e3:
            test_r["score"] = final_val["value"]
            res_df.append({"path": test, **test_r})
    res_df = pd.DataFrame.from_records(res_df)

    return res_df, scalars
