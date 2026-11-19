"""Cleaning, filtering, chronological splitting and 24-hour sessionisation.

Every step returns the transformed frame and appends a log entry so the
statistics report can show exactly how many rows each step removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_TRAJ_LEN, SESSION_GAP_HOURS, SPLIT_RATIOS, DatasetConfig

REQUIRED = ["user_raw", "venue_raw", "category", "local_time"]


def _log(log: list, step: str, before: int, after: int, **extra) -> None:
    log.append({"step": step, "rows_before": int(before), "rows_after": int(after),
                "removed": int(before - after), **extra})


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame, cfg: DatasetConfig, log: list) -> pd.DataFrame:
    df = df.copy()
    for col in ["user_raw", "venue_raw", "category_id", "category"]:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    if cfg.has_names:
        for col in ["name", "address"]:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})

    n = len(df)
    df = df.dropna(subset=REQUIRED)
    _log(log, "drop_missing_required_fields", n, len(df), fields=REQUIRED)

    n = len(df)
    valid = (df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
             & ~((df["lat"] == 0) & (df["lon"] == 0)))
    df = df[valid]
    _log(log, "drop_invalid_or_missing_coordinates", n, len(df))

    n = len(df)
    df = df.drop_duplicates(subset=["user_raw", "venue_raw", "local_time"])
    _log(log, "drop_duplicate_checkins(user,venue,time)", n, len(df))

    return df.sort_values(["user_raw", "local_time", "venue_raw"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Frequency filtering (single pass, POIs first, then users: LLM4POI/STHGCN)
# ---------------------------------------------------------------------------
def frequency_filter(df: pd.DataFrame, cfg: DatasetConfig, log: list) -> pd.DataFrame:
    n = len(df)
    poi_counts = df["venue_raw"].value_counts()
    df = df[df["venue_raw"].isin(poi_counts[poi_counts >= cfg.min_poi_checkins].index)]
    _log(log, f"drop_pois_with_<{cfg.min_poi_checkins}_checkins", n, len(df))

    n = len(df)
    user_counts = df["user_raw"].value_counts()
    df = df[df["user_raw"].isin(user_counts[user_counts >= cfg.min_user_checkins].index)]
    _log(log, f"drop_users_with_<{cfg.min_user_checkins}_checkins", n, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Chronological split on the global timeline
# ---------------------------------------------------------------------------
def time_cutoffs(times: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = np.sort(times.values)
    n = len(s)
    c1 = pd.Timestamp(s[int(np.floor(n * SPLIT_RATIOS[0]))])
    c2 = pd.Timestamp(s[int(np.floor(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])))])
    return c1, c2


def assign_split(df: pd.DataFrame, log: list) -> tuple[pd.DataFrame, dict]:
    c1, c2 = time_cutoffs(df["local_time"])
    df = df.copy()
    df["split"] = np.where(df["local_time"] < c1, "train",
                           np.where(df["local_time"] < c2, "val", "test"))
    _log(log, "assign_chronological_split", len(df), len(df),
         train_end=str(c1), val_end=str(c2))
    return df, {"train_before": str(c1), "val_before": str(c2)}


# ---------------------------------------------------------------------------
# 4. Sessionisation: new trajectory at a gap >= 24h (never crosses a split)
# ---------------------------------------------------------------------------
def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_raw", "local_time", "venue_raw"], kind="mergesort").reset_index(drop=True)
    gap = df["local_time"].diff()
    new_traj = ((df["user_raw"] != df["user_raw"].shift())
                | (df["split"] != df["split"].shift())
                | (gap >= pd.Timedelta(hours=SESSION_GAP_HOURS)))
    df["traj_key"] = new_traj.cumsum() - 1
    return df


def _drop_short(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    lengths = df.groupby("traj_key")["traj_key"].transform("size")
    return df[~(mask & (lengths < MIN_TRAJ_LEN))]


def build_trajectories(df: pd.DataFrame, log: list) -> pd.DataFrame:
    df = sessionize(df)

    # (a) training trajectories must have >= MIN_TRAJ_LEN check-ins
    n = len(df)
    df = _drop_short(df, df["split"] == "train")
    _log(log, f"drop_train_trajectories_shorter_than_{MIN_TRAJ_LEN}", n, len(df))

    # (b) val/test may only contain users and POIs seen in training
    train = df[df["split"] == "train"]
    known_users, known_pois = set(train["user_raw"]), set(train["venue_raw"])
    n = len(df)
    evalm = df["split"] != "train"
    df = df[~evalm | (df["user_raw"].isin(known_users) & df["venue_raw"].isin(known_pois))]
    _log(log, "drop_eval_checkins_with_user_or_poi_unseen_in_train", n, len(df))

    # Removing a middle check-in can leave a >=24h gap, so re-apply the session
    # rule. Training trajectories are unchanged (only whole ones were removed).
    df = sessionize(df)

    # (c) val/test trajectories must still have >= MIN_TRAJ_LEN check-ins
    n = len(df)
    df = _drop_short(df, df["split"] != "train")
    _log(log, f"drop_eval_trajectories_shorter_than_{MIN_TRAJ_LEN}", n, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Contiguous integer IDs (fitted on the training split only)
# ---------------------------------------------------------------------------
def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"]
    users = sorted(train["user_raw"].unique(), key=lambda x: (len(x), x))
    pois = sorted(train["venue_raw"].unique(), key=lambda x: (len(x), x))
    umap = {u: i for i, u in enumerate(users)}
    pmap = {p: i for i, p in enumerate(pois)}
    df = df.copy()
    df["user_id"] = df["user_raw"].map(umap).astype(int)
    df["poi_id"] = df["venue_raw"].map(pmap).astype(int)

    # trajectory ids: ordered by user, then time
    order = (df.groupby("traj_key").agg(user_id=("user_id", "first"), start=("local_time", "min"))
             .sort_values(["user_id", "start"]).reset_index())
    df["traj_id"] = df["traj_key"].map({k: i for i, k in enumerate(order["traj_key"])}).astype(int)
    df = df.drop(columns=["traj_key"]).sort_values(["user_id", "local_time", "poi_id"],
                                                   kind="mergesort").reset_index(drop=True)
    user_table = pd.DataFrame({"user_id": range(len(users)), "user_raw": users})
    return df, user_table, build_poi_table(df, pmap)


def build_poi_table(df: pd.DataFrame, pmap: dict) -> pd.DataFrame:
    cols = ["category_id", "category", "lat", "lon"] + [c for c in ["name", "address"] if c in df]
    first = df.sort_values("local_time").groupby("poi_id")[cols].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else pd.NA)
    first["lat"] = df.groupby("poi_id")["lat"].median()
    first["lon"] = df.groupby("poi_id")["lon"].median()
    counts = df[df["split"] == "train"].groupby("poi_id").size().rename("train_checkins")
    table = first.join(counts).reset_index()
    inv = {v: k for k, v in pmap.items()}
    table.insert(1, "venue_raw", table["poi_id"].map(inv))
    return table.sort_values("poi_id").reset_index(drop=True)
