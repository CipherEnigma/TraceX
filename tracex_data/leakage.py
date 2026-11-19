"""Target-leakage and protocol-integrity checks.

Each check returns {name, passed, detail}. The pipeline fails loudly if any
check fails, so a processed dataset on disk has always passed all of them.
"""
from __future__ import annotations

import pandas as pd

from .config import DEBUG_SAMPLE_SIZE, HISTORY_SPLITS, MIN_TRAJ_LEN, SAMPLE_SIZE, SESSION_GAP_HOURS
from .instances import make_samples


def _check(name: str, passed: bool, detail="") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_checks(df: pd.DataFrame, trajs: list[dict], instances: dict[str, list[dict]],
               samples: dict[str, list[dict]]) -> list[dict]:
    res = []
    by_split = {s: df[df["split"] == s] for s in ["train", "val", "test"]}

    # --- temporal separation of splits ---------------------------------------
    tmax, vmin = by_split["train"]["local_time"].max(), by_split["val"]["local_time"].min()
    vmax, smin = by_split["val"]["local_time"].max(), by_split["test"]["local_time"].min()
    res.append(_check("train_strictly_before_val", tmax < vmin, f"train max {tmax} < val min {vmin}"))
    res.append(_check("val_strictly_before_test", vmax < smin, f"val max {vmax} < test min {smin}"))

    # --- trajectories never cross splits / users -------------------------------
    per_traj = df.groupby("traj_id").agg(ns=("split", "nunique"), nu=("user_id", "nunique"))
    res.append(_check("each_trajectory_in_exactly_one_split", (per_traj["ns"] == 1).all()))
    res.append(_check("each_trajectory_has_one_user", (per_traj["nu"] == 1).all()))

    # --- session rule respected -----------------------------------------------
    d = df.sort_values(["traj_id", "local_time"])
    gaps = d.groupby("traj_id")["local_time"].diff().dropna()
    max_gap = gaps.max() if len(gaps) else pd.Timedelta(0)
    res.append(_check("no_within_trajectory_gap_>=24h", max_gap < pd.Timedelta(hours=SESSION_GAP_HOURS),
                      f"max within-trajectory gap = {max_gap}"))
    lens = df.groupby("traj_id").size()
    res.append(_check(f"all_trajectories_len_>={MIN_TRAJ_LEN}", (lens >= MIN_TRAJ_LEN).all(),
                      f"min length = {int(lens.min())}"))

    # --- eval users / POIs are known from training -----------------------------
    tu, tp = set(by_split["train"]["user_id"]), set(by_split["train"]["poi_id"])
    for s in ["val", "test"]:
        uu = set(by_split[s]["user_id"]) - tu
        pp = set(by_split[s]["poi_id"]) - tp
        res.append(_check(f"{s}_users_subset_of_train", not uu, f"{len(uu)} unseen users"))
        res.append(_check(f"{s}_pois_subset_of_train", not pp, f"{len(pp)} unseen POIs"))
    res.append(_check("poi_index_fitted_on_train_only",
                      sorted(tp) == list(range(len(tp))) and df["poi_id"].max() < len(tp)))

    # --- instance-level leakage --------------------------------------------------
    traj_by_id = {t["traj_id"]: t for t in trajs}
    for split, insts in instances.items():
        bad_hist_time, bad_hist_split, bad_prefix, bad_target, bad_self = 0, 0, 0, 0, 0
        for ins in insts:
            tgt_time = ins["target"]["time"]
            src = traj_by_id[ins["traj_id"]]
            if ins["target"] != src["checkins"][-1] or ins["prefix"] != src["checkins"][:-1]:
                bad_target += 1
            if any(c["time"] > tgt_time for c in ins["prefix"]):
                bad_prefix += 1
            if ins["traj_id"] in ins["history_traj_ids"]:
                bad_self += 1
            for hid in ins["history_traj_ids"]:
                h = traj_by_id[hid]
                if h["split"] not in HISTORY_SPLITS:
                    bad_hist_split += 1
                if h["end_time"] >= src["start_time"] or h["user_id"] != ins["user_id"]:
                    bad_hist_time += 1
        res.append(_check(f"{split}_target_is_last_checkin_and_not_in_prefix", bad_target == 0,
                          f"{bad_target} mismatches"))
        res.append(_check(f"{split}_prefix_not_after_target", bad_prefix == 0, f"{bad_prefix} violations"))
        res.append(_check(f"{split}_history_excludes_target_trajectory", bad_self == 0, f"{bad_self} violations"))
        res.append(_check(f"{split}_history_only_from_{'+'.join(HISTORY_SPLITS)}", bad_hist_split == 0,
                          f"{bad_hist_split} violations"))
        res.append(_check(f"{split}_history_strictly_before_and_same_user", bad_hist_time == 0,
                          f"{bad_hist_time} violations"))
        res.append(_check(f"{split}_every_instance_has_history", all(i["history_traj_ids"] for i in insts)))
        ids = [i["instance_id"] for i in insts]
        res.append(_check(f"{split}_instance_ids_unique", len(ids) == len(set(ids))))

    # --- reproducible samples ----------------------------------------------------
    again = make_samples(instances["test"])
    same = all([i["instance_id"] for i in again[k]] == [i["instance_id"] for i in v] for k, v in samples.items())
    res.append(_check("samples_reproducible_with_fixed_seed", same))
    test_ids = {i["instance_id"] for i in instances["test"]}
    main = samples[f"sample_test_{SAMPLE_SIZE}"]
    debug = samples[f"sample_test_debug{DEBUG_SAMPLE_SIZE}"]
    res.append(_check("sample_subset_of_test", all(i["instance_id"] in test_ids for i in main)))
    res.append(_check("debug_sample_subset_of_main_sample",
                      {i["instance_id"] for i in debug} <= {i["instance_id"] for i in main}))
    return res
