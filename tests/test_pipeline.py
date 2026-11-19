"""Unit tests for the TraceX data pipeline on small synthetic data.

Run:  python -m pytest tests -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from tracex_data.config import DatasetConfig
from tracex_data.instances import build_instances, build_trajectory_records, make_samples
from tracex_data.leakage import run_checks
from tracex_data.preprocess import (assign_split, build_trajectories, clean, frequency_filter,
                                    remap_ids, sessionize)

CFG = DatasetConfig(name="synth", source="foursquare", raw_path=None,
                    min_poi_checkins=1, min_user_checkins=1)
T0 = pd.Timestamp("2012-04-01 08:00:00")


def frame(rows):
    """rows: (user, venue, hours_after_T0)"""
    return pd.DataFrame({
        "user_raw": [str(u) for u, _, _ in rows],
        "venue_raw": [str(v) for _, v, _ in rows],
        "category_id": "c", "category": "Cafe", "lat": 40.7, "lon": -74.0,
        "local_time": [T0 + pd.Timedelta(hours=h) for _, _, h in rows],
    })


def test_session_gap_is_24h_inclusive():
    df = frame([(1, "a", 0), (1, "b", 23.99), (1, "c", 47.99), (1, "d", 72)])
    df["split"] = "train"
    out = sessionize(df)
    # a-b: 23.99h same; b-c: 24.0h -> new; c-d: 24.01h -> new
    assert out["traj_key"].tolist() == [0, 0, 1, 2]


def test_sessions_never_cross_users_or_splits():
    df = frame([(1, "a", 0), (2, "a", 1), (2, "b", 2)])
    df["split"] = ["train", "train", "test"]
    assert sessionize(df)["traj_key"].nunique() == 3


def test_clean_drops_missing_invalid_and_duplicates():
    df = frame([(1, "a", 0), (1, "a", 0), (1, "b", 1), (2, "c", 2)])
    df.loc[2, "lat"] = 999            # invalid coordinate
    df.loc[3, "venue_raw"] = ""       # missing venue
    out = clean(df, CFG, log := [])
    assert len(out) == 1
    assert [s["removed"] for s in log] == [1, 1, 1]


def test_frequency_filter_pois_then_users():
    cfg = DatasetConfig("s", "foursquare", None, min_poi_checkins=2, min_user_checkins=2)
    df = frame([(1, "a", 0), (1, "a", 1), (1, "b", 2), (2, "a", 3)])
    out = frequency_filter(df, cfg, [])
    assert set(out["venue_raw"]) == {"a"} and set(out["user_raw"]) == {"1"}


def synthetic_dataset():
    """3 users, ~4 months of daily 3-check-in trajectories over 6 POIs."""
    rows = []
    for u in range(3):
        for day in range(120):
            for k in range(3):
                rows.append((u, f"p{(day + k + u) % 6}", day * 24 + k * 2 + u * 0.1))
    rows.append((0, "never_in_train", 119 * 24 + 1))  # unseen POI inside a test trajectory
    return frame(rows)


def run_synthetic():
    df = clean(synthetic_dataset(), CFG, log := [])
    df = frequency_filter(df, CFG, log)
    df, _ = assign_split(df, log)
    df = build_trajectories(df, log)
    df, users, pois = remap_ids(df)
    trajs = build_trajectory_records(df, with_names=False)
    inst = {s: build_instances(trajs, "synth", s) for s in ["val", "test"]}
    return df, trajs, inst


def test_chronological_split_and_unseen_poi_removed():
    df, _, _ = run_synthetic()
    assert df.groupby("split")["local_time"].max()["train"] < df[df.split == "val"]["local_time"].min()
    assert df[df.split == "test"]["poi_id"].isin(df[df.split == "train"]["poi_id"]).all()
    assert "never_in_train" not in set(df.get("venue_raw", []))


def test_instances_target_history_and_revisit_flags():
    _, trajs, inst = run_synthetic()
    tmap = {t["traj_id"]: t for t in trajs}
    assert inst["test"], "expected test instances"
    for i in inst["test"]:
        traj = tmap[i["traj_id"]]
        assert i["target"] == traj["checkins"][-1]
        assert i["prefix"] == traj["checkins"][:-1]
        assert all(tmap[h]["split"] == "train" and tmap[h]["end_time"] < traj["start_time"]
                   for h in i["history_traj_ids"])
        hist = {c["poi_id"] for h in i["history_traj_ids"] for c in tmap[h]["checkins"]}
        assert i["target_in_history"] == (i["target"]["poi_id"] in hist)
        assert i["is_revisit"] == (i["target_in_history"] or i["target_in_prefix"])


def test_all_leakage_checks_pass_on_synthetic():
    df, trajs, inst = run_synthetic()
    samples = make_samples(inst["test"])
    failed = [c["name"] for c in run_checks(df, trajs, inst, samples) if not c["passed"]]
    assert failed == []


def test_leakage_check_catches_injected_future_history():
    df, trajs, inst = run_synthetic()
    bad = dict(inst["test"][0])
    future = [t["traj_id"] for t in trajs if t["split"] == "test" and t["user_id"] == bad["user_id"]]
    bad["history_traj_ids"] = bad["history_traj_ids"] + future
    inst["test"][0] = bad
    failed = {c["name"] for c in run_checks(df, trajs, inst, make_samples(inst["test"])) if not c["passed"]}
    assert "test_history_only_from_train" in failed


def test_samples_are_deterministic():
    _, _, inst = run_synthetic()
    a, b = make_samples(inst["test"]), make_samples(inst["test"])
    assert {k: [i["instance_id"] for i in v] for k, v in a.items()} == \
           {k: [i["instance_id"] for i in v] for k, v in b.items()}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
