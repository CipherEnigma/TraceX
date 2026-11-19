"""Turn the processed check-ins into trajectories and prediction instances.

An *instance* = one val/test trajectory. The input is the trajectory prefix
(all check-ins except the last) plus the user's history; the target is the
last check-in. This is the single task definition used by every method.
"""
from __future__ import annotations

import hashlib
import random

import pandas as pd

from .config import DEBUG_SAMPLE_SIZE, HISTORY_SPLITS, RANDOM_SEED, SAMPLE_SIZE


def checkin_record(row, with_names: bool) -> dict:
    t = row.local_time
    rec = {
        "poi_id": int(row.poi_id),
        "category": row.category,
        "time": t.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": int(t.hour),
        "weekday": int(t.weekday()),  # Monday = 0
    }
    if with_names:
        rec["name"] = None if pd.isna(row.name_) else str(row.name_)
    return rec


def build_trajectory_records(df: pd.DataFrame, with_names: bool) -> list[dict]:
    d = df.rename(columns={"name": "name_"}) if with_names else df
    trajs = []
    for traj_id, g in d.groupby("traj_id", sort=True):
        trajs.append({
            "traj_id": int(traj_id),
            "user_id": int(g["user_id"].iloc[0]),
            "split": g["split"].iloc[0],
            "start_time": g["local_time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": g["local_time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "checkins": [checkin_record(r, with_names) for r in g.itertuples(index=False)],
        })
    return trajs


def build_instances(trajs: list[dict], dataset: str, split: str) -> list[dict]:
    """One instance per trajectory of `split`. History = the user's trajectories
    in HISTORY_SPLITS that ended strictly before this trajectory started."""
    by_user: dict[int, list[dict]] = {}
    for t in trajs:
        if t["split"] in HISTORY_SPLITS:
            by_user.setdefault(t["user_id"], []).append(t)

    out = []
    for t in trajs:
        if t["split"] != split:
            continue
        hist = [h for h in by_user.get(t["user_id"], []) if h["end_time"] < t["start_time"]]
        if not hist:  # protocol: every eval user needs >=1 earlier training sequence
            continue
        prefix, target = t["checkins"][:-1], t["checkins"][-1]
        hist_pois = {c["poi_id"] for h in hist for c in h["checkins"]}
        prefix_pois = {c["poi_id"] for c in prefix}
        in_hist = target["poi_id"] in hist_pois
        in_prefix = target["poi_id"] in prefix_pois
        out.append({
            "instance_id": f"{dataset}-{split}-{t['traj_id']}",
            "dataset": dataset,
            "split": split,
            "user_id": t["user_id"],
            "traj_id": t["traj_id"],
            "history_traj_ids": [h["traj_id"] for h in hist],
            "prefix": prefix,
            "target": target,
            "target_in_history": in_hist,
            "target_in_prefix": in_prefix,
            "is_revisit": in_hist or in_prefix,   # E2 label (fixed definition)
            "n_history_trajs": len(hist),
            "n_history_checkins": sum(len(h["checkins"]) for h in hist),
        })
    return out


def sample_instances(instances: list[dict], k: int, seed: int = RANDOM_SEED) -> list[dict]:
    ordered = sorted(instances, key=lambda x: x["traj_id"])
    if len(ordered) <= k:
        return ordered
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(ordered)), k))
    return [ordered[i] for i in picked]


def make_samples(test_instances: list[dict]) -> dict[str, list[dict]]:
    main = sample_instances(test_instances, SAMPLE_SIZE)
    debug = sample_instances(main, DEBUG_SAMPLE_SIZE)  # debug set is a subset of the main sample
    return {f"sample_test_{SAMPLE_SIZE}": main, f"sample_test_debug{DEBUG_SAMPLE_SIZE}": debug}


def ids_fingerprint(instances: list[dict]) -> str:
    joined = "\n".join(i["instance_id"] for i in instances)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]
