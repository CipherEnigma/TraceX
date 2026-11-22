"""Stand-ins for Member 1's profile.py and Member 2's evidence.py (Member 3, task 3.1).

They return contract objects built from the real data with deliberately simple logic, so the LLM layer and the
dashboard can be built and demoed before the real modules land. `tracex_assist.api` switches to the real modules
automatically once they exist. Numbers produced here are NOT the assistant's results.
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import numpy as np

from tracex_data.api import load_dataset

from .contract import (Context, Evidence, Explore, Fact, Place, Profile, Suggestion, Visit, WEEKDAYS, routine_key,
                       time_bucket)

STUB = True


class _Places:
    def __init__(self, dataset: str):
        p = load_dataset(dataset).pois
        self._names = p["name"].tolist() if "name" in p.columns else [None] * len(p)
        self._p = p.reset_index(drop=True)
        self._row = {int(pid): i for i, pid in enumerate(self._p["poi_id"])}
        self._lat = np.radians(self._p["lat"].to_numpy(float))
        self._lon = np.radians(self._p["lon"].to_numpy(float))

    def get(self, poi_id: int) -> Place:
        r = self._p.iloc[self._row[int(poi_id)]]
        return Place(int(r["poi_id"]), str(r["category"]), self._names[self._row[int(poi_id)]], float(r["lat"]),
                     float(r["lon"]), int(r["train_checkins"]))

    def nearby(self, lat, lon, km, category=None):
        la, lo = math.radians(lat), math.radians(lon)
        h = np.sin((self._lat - la) / 2) ** 2 + math.cos(la) * np.cos(self._lat) * np.sin((self._lon - lo) / 2) ** 2
        d = 2 * 6371.0 * np.arcsin(np.sqrt(h))
        idx = np.where(d <= km)[0]
        if category:
            idx = idx[(self._p["category"].to_numpy()[idx] == category)]
        return sorted(((self.get(int(self._p["poi_id"].iat[i])), float(d[i])) for i in idx), key=lambda x: x[1])


@lru_cache(maxsize=4)
def place_index(dataset: str) -> _Places:
    return _Places(dataset)


@lru_cache(maxsize=4)
def _trips_by_user(dataset: str) -> dict:
    out: dict = {}
    for t in load_dataset(dataset).train_trajectories():
        out.setdefault(str(t["user_id"]), []).append(t)
    return out


def demo_users(dataset: str, n: int = 20) -> list[dict]:
    rows = []
    for uid, trips in _trips_by_user(dataset).items():
        cats = Counter(c["category"] for t in trips for c in t["checkins"])
        rows.append({"user_id": uid, "n_checkins": sum(len(t["checkins"]) for t in trips), "n_trips": len(trips),
                     "top_category": cats.most_common(1)[0][0]})
    return sorted(rows, key=lambda r: -r["n_checkins"])[:n]


def build_profile(dataset: str, user_id: str, history=None) -> Profile:
    trips_raw = history if history is not None else _trips_by_user(dataset)[str(user_id)]
    trips = [[Visit(int(c["poi_id"]), c["time"][:16], int(c["weekday"]), int(c["hour"]), str(t["traj_id"]), i)
              for i, c in enumerate(t["checkins"])] for t in trips_raw]
    visits = [v for t in trips for v in t]
    places = place_index(dataset)
    routine: dict = {}
    for v in visits:
        cat = places.get(v.poi_id).category
        routine.setdefault(routine_key(v.weekday, v.hour), Counter())[cat] += 1
    pts = [places.get(p) for p in {v.poi_id for v in visits}]
    return Profile(dataset, str(user_id), trips, Counter(v.poi_id for v in visits).most_common(),
                   {k: dict(c) for k, c in routine.items()}, len(visits), len(trips),
                   visits[0].time[:10] if visits else "", visits[-1].time[:10] if visits else "",
                   (float(np.mean([p.lat for p in pts])), float(np.mean([p.lon for p in pts]))) if pts else (0.0, 0.0))


def gather(profile: Profile, context: Context, places) -> Evidence:
    """Simple frequency + 'what came next after this place' evidence; no back-off, no fusion."""
    visits = [v for t in profile.trips for v in t]
    after = Counter()
    for t in profile.trips:
        for a, b in zip(t, t[1:]):
            if a.poi_id == context.current_poi:
                after[b.poi_id] += 1
    n_after = sum(after.values())
    freq = Counter(v.poi_id for v in visits)
    facts, habits = [], []
    for pid, c in after.most_common(3):
        facts.append(Fact("transition", pid, places.get(pid).category, c, n_after, "you, after this place, any day"))
    for pid, c in freq.most_common(5):
        f = Fact("frequency", pid, places.get(pid).category, c, len(visits), "you, all visits")
        facts.append(f)
        tr = [x for x in facts if x.kind == "transition" and x.poi_id == pid]
        habits.append(Suggestion(pid, c / len(visits) + (tr[0].share if tr else 0), tr + [f]))
    habits.sort(key=lambda s: -s.score)
    bucket = profile.routine.get(routine_key(context.weekday, context.hour), {})
    tot = sum(bucket.values())
    cats = [(c, n / tot) for c, n in Counter(bucket).most_common(3)] if tot else []
    for c, n in Counter(bucket).most_common(3):
        facts.append(Fact("time_habit", None, c, n, tot,
                          f"you, on {WEEKDAYS[context.weekday]} {time_bucket(context.hour)}s"))
    explore = []
    if context.current_poi is not None:
        here = places.get(context.current_poi)
        want = cats[0][0] if cats else (places.get(habits[0].poi_id).category if habits else here.category)
        seen = set(freq)
        for p, km in places.nearby(here.lat, here.lon, 2.0, want):
            if p.poi_id not in seen:
                explore.append(Explore(p.poi_id, round(km, 2), p.popularity, p.category))
        explore = sorted(explore, key=lambda e: -e.popularity)[:5]
    return Evidence(context, habits[:5], cats, explore, facts, {"acc1": None, "acc5": None, "n": 0, "stub": True},
                    thin=(n_after or len(visits)) < 5)
