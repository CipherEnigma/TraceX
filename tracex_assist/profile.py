"""User profiles, place index and demo users of the assistant (Member 1, tasks 1.1-1.3).

    from tracex_assist.profile import build_profile, demo_users, place_index
    places = place_index("nyc")
    places.nearby(40.73, -73.99, km=1.0, category="Coffee Shop")     # [(Place, km), ...], nearest first
    build_profile("nyc", "468")                                      # all of the user's train trips
    build_profile("nyc", "468", history=ds.history(inst))            # only these trips (offline evaluation)

    python -m tracex_assist.profile                                  # writes data/assistant/demo_users.json

Leakage: a profile reads only the trips it is given (train trips by default) and popularity counts train check-ins
only. A POI's category and coordinates are static metadata from the pipeline's pois.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from statistics import median

import numpy as np

from tracex_data.api import load_dataset
from tracex_data.config import DATASETS, ROOT

try:  # contract.py is the newer name of types.py; only one of the two may be on the current branch
    from .contract import Place, Profile, Visit, routine_key, time_bucket
except ModuleNotFoundError:
    from .types import Place, Profile, Visit, routine_key, time_bucket

EARTH_KM = 6371.0
DEMO_PATH = ROOT / "data" / "assistant" / "demo_users.json"


@lru_cache(maxsize=4)
def _dataset(name: str):
    return load_dataset(name)


# ---------------------------------------------------------------------------
# 1.1 Place index
# ---------------------------------------------------------------------------
class Places:
    """Implements the contract's PlaceIndex: lookup by id, radius search on a lat/lon grid."""

    CELL = 0.01  # degrees, about 1 km

    def __init__(self, pois):
        names = list(pois["name"]) if "name" in pois.columns else [None] * len(pois)
        self._places = [
            Place(int(pid), str(cat), name if isinstance(name, str) else None, float(lat), float(lon), int(n))
            for pid, cat, name, lat, lon, n in zip(pois["poi_id"], pois["category"], names, pois["lat"], pois["lon"],
                                                   pois["train_checkins"])]
        self._by_id = {p.poi_id: p for p in self._places}
        self._lat = np.radians([p.lat for p in self._places])
        self._lon = np.radians([p.lon for p in self._places])
        self._grid = defaultdict(list)
        for i, p in enumerate(self._places):
            self._grid[self._cell(p.lat), self._cell(p.lon)].append(i)

    def _cell(self, degrees: float) -> int:
        return math.floor(degrees / self.CELL)

    def get(self, poi_id: int) -> Place:
        return self._by_id[int(poi_id)]

    def _candidates(self, lat: float, lon: float, km: float):
        """Rows in the grid cells that cover the circle (a superset of the answer)."""
        r = km / EARTH_KM
        s, c = math.sin(r), math.cos(math.radians(lat))
        if r >= math.pi / 2 or s >= c:  # the circle reaches a pole
            return range(len(self._places))
        dlat, dlon = math.degrees(r), math.degrees(math.asin(s / c))
        if lon - dlon < -180 or lon + dlon > 180:  # crosses the antimeridian
            return range(len(self._places))
        rows = range(self._cell(lat - dlat), self._cell(lat + dlat) + 1)
        cols = range(self._cell(lon - dlon), self._cell(lon + dlon) + 1)
        if len(rows) * len(cols) > len(self._places):  # more cells to visit than places to check
            return range(len(self._places))
        return [i for a in rows for b in cols for i in self._grid.get((a, b), ())]

    def nearby(self, lat: float, lon: float, km: float, category: str | None = None) -> list[tuple[Place, float]]:
        """Places within `km` (the place at the centre included), nearest first."""
        idx = [i for i in self._candidates(lat, lon, km) if category is None or self._places[i].category == category]
        if not idx:
            return []
        la, lo = math.radians(lat), math.radians(lon)
        h = (np.sin((self._lat[idx] - la) / 2) ** 2
             + math.cos(la) * np.cos(self._lat[idx]) * np.sin((self._lon[idx] - lo) / 2) ** 2)
        dist = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        hits = [(self._places[i], float(d)) for i, d in zip(idx, dist) if d <= km]
        return sorted(hits, key=lambda x: (x[1], x[0].poi_id))


@lru_cache(maxsize=4)
def place_index(dataset: str) -> Places:
    return Places(_dataset(dataset).pois)


# ---------------------------------------------------------------------------
# 1.2 Profile
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _train_by_user(dataset: str) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for t in _dataset(dataset).train_trajectories():
        out[str(t["user_id"])].append(t)
    return dict(out)


def build_profile(dataset: str, user_id: str, history: list[dict] | None = None) -> Profile:
    """Profile from `history` (default: all of the user's train trips). An empty history gives an empty profile.

    top_places breaks ties by POI id, as the baselines do. home is the centroid of the distinct places, and
    (0.0, 0.0) if there are none.
    """
    uid = str(user_id)
    if history is None:
        try:
            history = _train_by_user(dataset)[uid]
        except KeyError:
            raise KeyError(f"user {uid!r} has no train trips in {dataset}") from None
    elif any(str(t["user_id"]) != uid for t in history):
        raise ValueError(f"history holds trips of a user other than {uid!r}")

    raw = sorted((t for t in history if t["checkins"]), key=lambda t: t["checkins"][0]["time"])
    trips = [[Visit(int(c["poi_id"]), c["time"][:16], int(c["weekday"]), int(c["hour"]), str(t["traj_id"]), i)
              for i, c in enumerate(t["checkins"])] for t in raw]
    visits = [v for t in trips for v in t]
    counts = Counter(v.poi_id for v in visits)

    places = place_index(dataset)
    routine = defaultdict(Counter)
    for v in visits:
        # category from the index, so it equals what Place and nearby(category=...) use
        routine[routine_key(v.weekday, v.hour)][places.get(v.poi_id).category] += 1
    points = [places.get(p) for p in sorted(counts)]
    days = [v.time[:10] for v in visits]

    return Profile(
        dataset=dataset,
        user_id=uid,
        trips=trips,
        top_places=sorted(counts.items(), key=lambda x: (-x[1], x[0])),
        routine={key: dict(c.most_common()) for key, c in routine.items()},
        n_checkins=len(visits),
        n_trips=len(trips),
        first_day=min(days, default=""),
        last_day=max(days, default=""),
        home=(float(np.mean([p.lat for p in points])), float(np.mean([p.lon for p in points]))) if points
        else (0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# 1.3 Demo users
# ---------------------------------------------------------------------------
N_PERSONAS = 5
MIN_TOTAL, MIN_SHARE = 5, 0.5  # a "clear routine" is a majority of 5+ visits (fewer is thin evidence)
MAX_RANK = 5  # the place they go next is also one of their top places, so habit and transition evidence agree
MIN_GAP_H, MAX_GAP_H = 1 / 6, 6.0  # median gap: not a double check-in, and still "I just left X"
NEW_RADII_KM, MIN_NEW = (1.0, 2.0, 5.0), 3  # "something new" needs this many unvisited places (radii of task 2.3)

# Never shown as a persona's routine: homes, workplaces, schools, health, worship, and places that are not venues.
NOT_FOR_DEMO = re.compile(
    r"home \(private\)|residen|housing|fraternity|sorority|office|school|medical|hospital|doctor|dentist|pharmacy|"
    r"drugstore|church|mosque|synagogue|spiritual|temple|shrine|cemetery|funeral|military|police|recruiting|"
    r"financial|embassy|gun range|smoke shop|vape|moving target|neighborhood|\broad\b|intersection|building|"
    r"^city$|bathroom", re.I)
GYM = re.compile(r"\bgym\b|fitness", re.I)
EAT = re.compile(
    r"restaurant|food|burger|pizza|sushi|ramen|noodle|diner|deli /|bakery|sandwich|steakhouse|bbq|taco|burrito|"
    r"chicken|donut|dessert|ice cream|bagel|breakfast|coffee|caf[eé]$|tea room|dumpling|bistro|gastropub", re.I)
COFFEE = re.compile(r"coffee|^caf[eé]$", re.I)
OUTING = re.compile(
    r"\bbar\b|\bpub\b|lounge|gym|fitness|\bpark\b|movie|multiplex|\bmall\b|store|shop|market|museum|arcade|"
    r"bowling|theater|music venue|stadium|spa\b|garden|beach|plaza|library|karaoke", re.I)


def _wilson_low(count: int, total: int, z: float = 1.96) -> float:
    """Lower bound of a share, so that 19 of 20 ranks above 3 of 3."""
    f, z2 = count / total, z * z
    return (f + z2 / (2 * total) - z * math.sqrt((f * (1 - f) + z2 / (4 * total)) / total)) / (1 + z2 / total)


def _next_steps(trips: list[dict]) -> dict[int, list[tuple[dict, dict]]]:
    out = defaultdict(list)
    for t in trips:
        for a, b in zip(t["checkins"], t["checkins"][1:]):
            out[a["poi_id"]].append((a, b))
    return out


def _hours(a: dict, b: dict) -> float:
    return (datetime.fromisoformat(b["time"]) - datetime.fromisoformat(a["time"])).total_seconds() / 3600


def _usual_time(checkins: list[dict]) -> tuple[int, int]:
    """Most common (weekday, time bucket), then the most common hour inside it."""
    slots = Counter((c["weekday"], time_bucket(c["hour"])) for c in checkins)
    day, bucket = min(slots, key=lambda s: (-slots[s], s))
    hours = Counter(c["hour"] for c in checkins if c["weekday"] == day and time_bucket(c["hour"]) == bucket)
    return day, min(hours, key=lambda h: (-hours[h], h))


def _routines(dataset: str) -> list[dict]:
    """Every "after A you usually go to B" that can be shown in a demo, best first."""
    places, out = place_index(dataset), []
    for uid, trips in _train_by_user(dataset).items():
        visits = Counter(c["poi_id"] for t in trips for c in t["checkins"])
        usual = sorted(visits, key=lambda p: (-visits[p], p))[:MAX_RANK]
        for here, pairs in _next_steps(trips).items():
            counts, total = Counter(b["poi_id"] for _, b in pairs), len(pairs)
            then = min(counts, key=lambda p: (-counts[p], p))
            count = counts[then]
            if then == here or then not in usual or total < MIN_TOTAL or count / total <= MIN_SHARE:
                continue
            a, b = places.get(here), places.get(then)
            if NOT_FOR_DEMO.search(a.category) or NOT_FOR_DEMO.search(b.category):
                continue
            eats = bool(EAT.search(b.category))
            if not eats and not OUTING.search(b.category):  # must be an answer to "where should I go?"
                continue
            taken = [(x, y) for x, y in pairs if y["poi_id"] == then]
            gap = median(_hours(x, y) for x, y in taken)
            if not MIN_GAP_H <= gap <= MAX_GAP_H:
                continue
            day, hour = _usual_time([y for _, y in taken])
            out.append({
                "user_id": uid,
                "from_poi": here, "from_category": a.category, "from_name": a.name,
                "to_poi": then, "to_category": b.category, "to_name": b.name,
                "count": count, "total": total, "median_gap_hours": round(gap, 2),
                "weekday": day, "hour": hour,
                "_order": (0 if eats and GYM.search(a.category) else 1 if eats else 2,
                           -_wilson_low(count, total), int(uid), here),
            })
    return sorted(out, key=lambda r: r["_order"])


@lru_cache(maxsize=4)
def _personas(dataset: str) -> tuple[dict, ...]:
    """Up to five users: gym -> food first, then -> food, then other outings; varied starting places and users."""
    picked, users, starts = [], set(), set()
    candidates = _routines(dataset)
    for varied in (True, False):  # second pass only if there are too few different starting categories
        for r in candidates:
            if len(picked) == N_PERSONAS:
                break
            if r["user_id"] in users or (varied and r["from_category"] in starts):
                continue
            picked.append({k: v for k, v in r.items() if k != "_order"})
            users.add(r["user_id"])
            starts.add(r["from_category"])
    return tuple(picked)


@lru_cache(maxsize=4)
def _ranked_users(dataset: str) -> tuple[dict, ...]:
    """All train users by richness: mean percentile of check-ins, trips and distinct places."""
    places, rows = place_index(dataset), []
    for uid, trips in _train_by_user(dataset).items():
        ids = [c["poi_id"] for t in trips for c in t["checkins"]]
        cats = Counter(places.get(p).category for p in ids)
        rows.append({"user_id": uid, "n_checkins": len(ids), "n_trips": len(trips), "n_places": len(set(ids)),
                     "top_category": min(cats, key=lambda c: (-cats[c], c))})
    score = np.zeros(len(rows))
    for key in ("n_checkins", "n_trips", "n_places"):
        values = np.array([r[key] for r in rows])
        score += np.searchsorted(np.sort(values), values, side="right")
    order = sorted(range(len(rows)), key=lambda i: (-score[i], int(rows[i]["user_id"])))
    return tuple(rows[i] for i in order)


def demo_users(dataset: str, n: int = 20) -> list[dict]:
    """The personas first, then the richest users. `persona` holds the routine to demo, or None."""
    ranked = _ranked_users(dataset)
    by_id = {r["user_id"]: r for r in ranked}
    personas = {p["user_id"]: p for p in _personas(dataset)}
    ordered = [by_id[u] for u in personas] + [r for r in ranked if r["user_id"] not in personas]
    return [{**r, "persona": dict(personas[r["user_id"]]) if r["user_id"] in personas else None}
            for r in ordered[:n]]


def _scenarios(dataset: str) -> dict:
    """The three rehearsal scenarios.

    Questions never name the current place: the intent parser would read it as the kind of place asked for.
    """
    places, by_user, personas = place_index(dataset), _train_by_user(dataset), _personas(dataset)
    out = {}

    def scenario(question, r, **more):
        return {"question": question, "user_id": r["user_id"], "current_poi": r["from_poi"],
                "weekday": r["weekday"], "hour": r["hour"], **more}

    if personas:
        p = personas[0]
        out["habit"] = scenario("Where should I go next?", p,
                                shows={k: p[k] for k in ("to_poi", "count", "total")})

    others = personas[1:] + personas[:1]  # preferably a different persona than the habit
    for km, p in ((km, p) for km in NEW_RADII_KM for p in others):
        seen = {c["poi_id"] for t in by_user[p["user_id"]] for c in t["checkins"]}
        here = places.get(p["from_poi"])
        unvisited = [x for x, _ in places.nearby(here.lat, here.lon, km, p["to_category"])
                     if x.poi_id not in seen]
        if len(unvisited) >= MIN_NEW:
            cat = p["to_category"]
            what = ("for coffee" if COFFEE.search(cat) else "to eat" if EAT.search(cat) else "nearby")
            out["something_new"] = scenario(f"Suggest somewhere new {what}.", p, want_new=True,
                                            shows={"category": cat, "km": km, "unvisited_nearby": len(unvisited)})
            break

    sizes = {uid: sum(len(t["checkins"]) for t in trips) for uid, trips in by_user.items()}
    # fewer than 5 visits in total is thin at every back-off level of the user
    for uid in sorted((u for u, size in sizes.items() if size < MIN_TOTAL), key=lambda u: (-sizes[u], int(u))):
        checkins = [c for t in by_user[uid] for c in t["checkins"]]
        cats = [places.get(c["poi_id"]).category for c in checkins]
        if any(NOT_FOR_DEMO.search(c) for c in cats):
            continue
        a, b = by_user[uid][-1]["checkins"][-2:]
        then = places.get(b["poi_id"]).category
        if a["poi_id"] != b["poi_id"] and (EAT.search(then) or OUTING.search(then)):
            out["thin_evidence"] = scenario(
                "Where should I go next?",
                {"user_id": uid, "from_poi": a["poi_id"], "weekday": b["weekday"], "hour": b["hour"]},
                shows={"n_checkins": len(checkins)})
            break
    return out


def write_demo_users(datasets, path=DEMO_PATH) -> dict:
    out = {"generated_by": "python -m tracex_assist.profile", "datasets": {}}
    for d in datasets:
        out["datasets"][d] = {"personas": demo_users(d, len(_personas(d))), "scenarios": _scenarios(d)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_assist.profile", description="Write the demo personas and scenarios.")
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    a = ap.parse_args(argv)
    out = write_demo_users(a.dataset)
    for d, v in out["datasets"].items():
        print(f"== {d}: {len(v['personas'])} personas, scenarios: {', '.join(v['scenarios'])}")
        for row in v["personas"]:
            p = row["persona"]
            print(f"  user {row['user_id']:>5}: {p['from_category']} #{p['from_poi']} -> {p['to_category']} "
                  f"#{p['to_poi']}, {p['count']} of {p['total']}".encode("ascii", "replace").decode())
    print(f"wrote {DEMO_PATH}")


if __name__ == "__main__":
    main()
