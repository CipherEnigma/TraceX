"""Load each raw dataset into one unified schema.

Unified columns:
    user_raw, venue_raw, category_id, category, lat, lon, local_time
    (+ name, address for datasets with venue names)
`local_time` is a timezone-naive pandas Timestamp in the venue's local time.
"""
from __future__ import annotations

import pandas as pd

from .config import DatasetConfig

FOURSQUARE_COLS = [
    "user_raw", "venue_raw", "category_id", "category",
    "lat", "lon", "tz_offset_min", "utc_time",
]


def load_foursquare(path) -> tuple[pd.DataFrame, dict]:
    # The TSMC2014 files contain a few non-UTF-8 bytes in category names.
    df = pd.read_csv(path, sep="\t", header=None, names=FOURSQUARE_COLS,
                     encoding="latin-1", dtype=str, keep_default_na=False)
    n_raw = len(df)
    utc = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y",
                         errors="coerce", utc=True)
    offset = pd.to_numeric(df["tz_offset_min"], errors="coerce")
    df["local_time"] = (utc + pd.to_timedelta(offset, unit="m")).dt.tz_localize(None)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.drop(columns=["tz_offset_min", "utc_time"])
    return df, {"raw_rows": n_raw}


def load_massive_steps(path, timezone: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype={"user_id": str, "venue_id": str, "trail_id": str})
    # Massive-STEPS timestamps are UTC (not documented; verified from the data:
    # raw Tokyo bar check-ins peak at 12:00 = 21:00 JST). Convert to local time.
    utc = pd.to_datetime(raw["timestamp"], errors="coerce").dt.tz_localize("UTC")
    local = utc.dt.tz_convert(timezone).dt.tz_localize(None)
    df = pd.DataFrame({
        "user_raw": raw["user_id"],
        "venue_raw": raw["venue_id"],
        "category_id": raw["venue_category_id"].astype(str),
        "category": raw["venue_category"].astype(str),
        "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
        "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
        "local_time": local,
        "name": raw["name"],
        "address": raw["address"],
        "source_trail_id": raw["trail_id"],
    })
    return df, {"raw_rows": len(raw)}


def load_raw(cfg: DatasetConfig) -> tuple[pd.DataFrame, dict]:
    if cfg.source == "foursquare":
        return load_foursquare(cfg.raw_path)
    if cfg.source == "massive_steps":
        return load_massive_steps(cfg.raw_path, cfg.extra["timezone"])
    raise ValueError(f"unknown source {cfg.source}")
