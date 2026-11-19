"""Dataset statistics and POI-coverage analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DatasetConfig

# Reference sizes after standard filtering, from the project brief (LLM4POI / Nishida et al.)
PUBLISHED = {"nyc": {"users": 1048, "pois": 4981}, "tky": {"users": 2282, "pois": 7833}}


def _desc(s: pd.Series) -> dict:
    s = pd.Series(s)
    return {"mean": round(float(s.mean()), 2), "median": float(s.median()),
            "min": int(s.min()), "max": int(s.max()), "p90": float(s.quantile(0.9))}


def _rate(xs) -> float:
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else float("nan")


def instance_stats(insts: list[dict]) -> dict:
    return {
        "n": len(insts),
        "revisit_rate": _rate(i["is_revisit"] for i in insts),
        "target_in_history_rate": _rate(i["target_in_history"] for i in insts),
        "target_in_prefix_rate": _rate(i["target_in_prefix"] for i in insts),
        "new_poi_rate": _rate(not i["is_revisit"] for i in insts),
        "prefix_len": _desc([len(i["prefix"]) for i in insts]) if insts else {},
        "history_trajs": _desc([i["n_history_trajs"] for i in insts]) if insts else {},
        "history_checkins": _desc([i["n_history_checkins"] for i in insts]) if insts else {},
    }


def compute_stats(cfg: DatasetConfig, raw_meta: dict, log: list, df: pd.DataFrame,
                  pois: pd.DataFrame, instances: dict, samples: dict, cutoffs: dict) -> dict:
    lens = df.groupby("traj_id").size()
    per_user = df.groupby("user_id").size()
    trajs_per_user = df.groupby("user_id")["traj_id"].nunique()
    train = df[df["split"] == "train"]
    poi_freq = train["poi_id"].value_counts().sort_values(ascending=False)
    top1pct = max(1, int(round(len(poi_freq) * 0.01)))

    st = {
        "dataset": cfg.name,
        "description": cfg.description,
        "filters": {"min_poi_checkins": cfg.min_poi_checkins, "min_user_checkins": cfg.min_user_checkins},
        "raw_rows": raw_meta["raw_rows"],
        "pipeline_log": log,
        "split_cutoffs": cutoffs,
        "overall": {
            "users": int(df["user_id"].nunique()),
            "pois": int(df["poi_id"].nunique()),
            "categories": int(df["category"].nunique()),
            "checkins": int(len(df)),
            "trajectories": int(df["traj_id"].nunique()),
            "time_span": [str(df["local_time"].min()), str(df["local_time"].max())],
            "checkins_per_user": _desc(per_user),
            "trajectories_per_user": _desc(trajs_per_user),
            "trajectory_length": _desc(lens),
        },
        "per_split": {
            s: {"checkins": int((g := df[df["split"] == s]).shape[0]),
                "trajectories": int(g["traj_id"].nunique()),
                "users": int(g["user_id"].nunique()),
                "pois": int(g["poi_id"].nunique())}
            for s in ["train", "val", "test"]
        },
        "poi_coverage": {
            "pois_in_train": int(len(poi_freq)),
            "train_checkins_per_poi": _desc(poi_freq),
            "share_of_train_checkins_on_top_1pct_pois": round(float(poi_freq.iloc[:top1pct].sum() / poi_freq.sum()), 4),
            "share_of_pois_with_<=5_train_checkins": round(float((poi_freq <= 5).mean()), 4),
            "test_pois_seen_in_train": round(float(df[df["split"] == "test"]["poi_id"].isin(poi_freq.index).mean()), 4),
            "test_target_pois_distinct": len({i["target"]["poi_id"] for i in instances["test"]}),
        },
        "instances": {k: instance_stats(v) for k, v in {**instances, **samples}.items()},
    }
    if cfg.name in PUBLISHED:
        st["published_reference"] = PUBLISHED[cfg.name]
    if cfg.has_names:
        named = pois["name"].notna()
        st["name_coverage"] = {
            "pois_with_name": round(float(named.mean()), 4),
            "pois_with_address": round(float(pois["address"].notna().mean()), 4),
            "checkins_with_name": round(float(df["name"].notna().mean()), 4),
            "test_targets_with_name": _rate(i["target"].get("name") is not None for i in instances["test"]),
            "distinct_names": int(pois["name"].nunique()),
            "pois_sharing_a_name_with_another_poi": round(float(pois["name"].duplicated(keep=False)[named].mean()), 4),
        }
    return st
