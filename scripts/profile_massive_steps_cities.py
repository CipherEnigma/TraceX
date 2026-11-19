"""Choose the Massive-STEPS city for E3 (name test).

The brief: "Tokyo is the natural pick ... Check per-user history length and
name coverage first; if either is poor, use another city from the same release."

Usage:  python scripts/profile_massive_steps_cities.py [City ...]
Writes reports/massive_steps_city_profile.csv and .md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracex_data.config import REPORTS_DIR  # noqa: E402
from tracex_data.download import download_massive_steps  # noqa: E402

DEFAULT = ["Tokyo", "New-York", "Istanbul", "Jakarta", "Sao-Paulo"]


def profile(city: str) -> dict:
    df = pd.read_csv(download_massive_steps(city), dtype={"user_id": str, "venue_id": str})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])
    gap = df["timestamp"].diff()
    sess = ((df["user_id"] != df["user_id"].shift()) | (gap >= pd.Timedelta(hours=24))).cumsum()
    per_user = df.groupby("user_id").size()
    sess_per_user = sess.groupby(df["user_id"]).nunique()
    venues = df.drop_duplicates("venue_id")
    return {
        "city": city,
        "checkins": len(df),
        "users": df["user_id"].nunique(),
        "pois": df["venue_id"].nunique(),
        "period": f"{df['timestamp'].min():%Y-%m} to {df['timestamp'].max():%Y-%m}",
        "median_checkins_per_user": float(per_user.median()),
        "median_24h_sessions_per_user": float(sess_per_user.median()),
        "users_with_>=3_sessions": round(float((sess_per_user >= 3).mean()), 3),
        "checkins_with_name": round(float(df["name"].notna().mean()), 3),
        "pois_with_name": round(float(venues["name"].notna().mean()), 3),
        "pois_with_address": round(float(venues["address"].notna().mean()), 3),
    }


def main(cities: list[str]) -> None:
    rows = [profile(c) for c in cities]
    t = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    t.to_csv(REPORTS_DIR / "massive_steps_city_profile.csv", index=False)
    (REPORTS_DIR / "massive_steps_city_profile.md").write_text(t.to_markdown(index=False) + "\n", encoding="utf-8")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
