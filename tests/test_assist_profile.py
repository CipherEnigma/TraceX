"""Tests for profiles, the place index and demo users (Member 1, task 1.5) on synthetic data.

Run:  python -m pytest tests/test_assist_profile.py -q
"""
from __future__ import annotations

import copy
import json
import math
import random

import pandas as pd
import pytest

from tracex_assist import profile as P


def check_in(poi, time):
    ts = pd.Timestamp(time)
    return {"poi_id": poi, "time": ts.strftime("%Y-%m-%d %H:%M:%S"), "hour": ts.hour, "weekday": ts.weekday(),
            "category": "ignored: the index decides"}


def trip(traj_id, user, *steps):
    return {"traj_id": traj_id, "user_id": user, "split": "train", "checkins": [check_in(*s) for s in steps]}


def poi_table(rows, names=None):
    df = pd.DataFrame(rows, columns=["poi_id", "category", "lat", "lon", "train_checkins"])
    if names is not None:
        df["name"] = names
    return df


POIS = [(1, "Gym", 40.000, -74.000, 50), (2, "Burger Joint", 40.001, -74.000, 90),
        (3, "Coffee Shop", 40.002, -74.000, 30), (4, "Home (private)", 40.003, -74.000, 7),
        (5, "Burger Joint", 40.004, -74.000, 12), (6, "Burger Joint", 40.005, -74.000, 11),
        (7, "Burger Joint", 40.006, -74.000, 10), (8, "Coffee Shop", 40.500, -74.000, 4)]

USER7 = [trip(2, 7, (3, "2013-01-07 02:15:40"), (1, "2013-01-07 08:00:00")),            # Monday
         trip(1, 7, (1, "2013-01-04 18:05:59"), (2, "2013-01-04 19:00:00"), (1, "2013-01-04 23:30:00"))]  # Friday


class FakeData:
    def __init__(self, trips, pois=POIS):
        self.pois, self._trips = poi_table(pois), trips

    def train_trajectories(self):
        return self._trips


def clear_caches():
    for f in (P.place_index, P._train_by_user, P._personas, P._ranked_users):
        f.cache_clear()


@pytest.fixture
def use(monkeypatch):
    """use(trips) makes every dataset name resolve to a synthetic dataset with these trips."""
    def install(trips):
        data = FakeData(trips)
        monkeypatch.setattr(P, "_dataset", lambda name: data)
        clear_caches()
    yield install
    clear_caches()


# ---------------------------------------------------------------- time buckets
def test_time_buckets_wrap_midnight_and_cover_the_day():
    expected = (["night"] * 5 + ["morning"] * 6 + ["midday"] * 3 + ["afternoon"] * 3 + ["evening"] * 4
                + ["night"] * 3)
    assert [P.time_bucket(h) for h in range(24)] == expected


# ---------------------------------------------------------------- profile
def test_profile_of_a_synthetic_user(use):
    use(USER7 + [trip(3, 9, (5, "2013-01-01 10:00"), (6, "2013-01-01 11:00"))])
    p = P.build_profile("synth", "7")
    assert (p.dataset, p.user_id, p.n_checkins, p.n_trips) == ("synth", "7", 5, 2)
    assert [[v.poi_id for v in t] for t in p.trips] == [[1, 2, 1], [3, 1]]  # oldest trip first
    v = p.trips[0][0]
    assert (v.time, v.weekday, v.hour, v.trip, v.pos) == ("2013-01-04 18:05", 4, 18, "1", 0)
    assert [x.pos for x in p.trips[0]] == [0, 1, 2]
    assert p.top_places == [(1, 3), (2, 1), (3, 1)]  # ties by POI id
    assert (p.first_day, p.last_day) == ("2013-01-04", "2013-01-07")
    assert p.home == pytest.approx((40.001, -74.0))  # centroid of the distinct places, not of the visits


def test_routine_by_weekday_and_bucket_puts_small_hours_in_night(use):
    use(USER7)
    assert P.build_profile("synth", "7").routine == {
        "Fri|evening": {"Gym": 1, "Burger Joint": 1}, "Fri|night": {"Gym": 1},
        "Mon|night": {"Coffee Shop": 1}, "Mon|morning": {"Gym": 1}}


def test_user_id_may_be_an_int(use):
    use(USER7)
    assert P.build_profile("synth", 7) == P.build_profile("synth", "7")


def test_history_is_the_only_source(use):
    use(USER7)
    p = P.build_profile("synth", "7", history=[USER7[1]])
    assert (p.n_checkins, p.n_trips, p.top_places) == (3, 1, [(1, 2), (2, 1)])
    assert not any(k.startswith("Mon") for k in p.routine)
    assert p.last_day == "2013-01-04" and p.home == pytest.approx((40.0005, -74.0))


def test_history_does_not_read_the_train_trips(use, monkeypatch):
    use(USER7)

    def forbidden(dataset):
        raise AssertionError("train trips were read although a history was given")

    monkeypatch.setattr(P, "_train_by_user", forbidden)
    assert P.build_profile("synth", "7", history=[USER7[0]]).n_trips == 1
    monkeypatch.undo()


def test_empty_history_is_empty_and_not_the_default(use):
    use(USER7)
    p = P.build_profile("synth", "7", history=[])
    assert (p.trips, p.top_places, p.routine, p.n_checkins, p.first_day, p.home) == ([], [], {}, 0, "", (0.0, 0.0))


def test_history_is_not_modified(use):
    use(USER7)
    given = copy.deepcopy(USER7)
    P.build_profile("synth", "7", history=given)
    assert given == USER7


def test_wrong_or_unknown_user_is_refused(use):
    use(USER7)
    with pytest.raises(ValueError):
        P.build_profile("synth", "8", history=USER7)
    with pytest.raises(KeyError):
        P.build_profile("synth", "8")


# ---------------------------------------------------------------- place index
def haversine(a, b, c, d):
    a, b, c, d = map(math.radians, (a, b, c, d))
    h = math.sin((c - a) / 2) ** 2 + math.cos(a) * math.cos(c) * math.sin((d - b) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def test_get_and_missing_names():
    idx = P.Places(poi_table(POIS[:3], names=["ジム", None, float("nan")]))
    first = idx.get(1)
    assert (first.category, first.name, first.popularity) == ("Gym", "ジム", 50)
    assert idx.get(2).name is None and idx.get(3).name is None
    assert P.Places(poi_table(POIS)).get(1).name is None  # no name column (NYC, TKY)
    with pytest.raises(KeyError):
        idx.get(99)


def test_nearby_order_category_and_radius():
    idx = P.Places(poi_table(POIS))
    hits = idx.nearby(40.0, -74.0, 0.35)
    assert [p.poi_id for p, _ in hits] == [1, 2, 3, 4]
    assert hits[0][1] == 0 and hits[1][1] == pytest.approx(0.1112, abs=1e-4)
    assert [p.poi_id for p, _ in idx.nearby(40.0, -74.0, 1.0, "Burger Joint")] == [2, 5, 6, 7]
    assert idx.nearby(40.0, -74.0, 1.0, "No such category") == []
    assert idx.nearby(10.0, 10.0, 1.0) == []


@pytest.mark.parametrize("lat0,lon0,spread", [
    (40.73, -73.99, 0.08),   # New York
    (35.66, 139.70, 0.08),   # Tokyo
    (0.0, 0.0, 0.08),        # cells with negative indices
    (-33.9, 151.2, 0.08),
    (70.0, 20.0, 0.15),      # narrow cells in longitude
    (10.0, 180.0, 0.08),     # antimeridian
    (89.97, 10.0, 0.03),     # next to the pole
])
def test_nearby_equals_brute_force(lat0, lon0, spread):
    rng = random.Random(42)

    def point():
        lat = min(90.0, lat0 + rng.uniform(-spread, spread))
        return lat, (lon0 + rng.uniform(-spread, spread) + 180) % 360 - 180

    rows = [(i, rng.choice("AB"), *point(), 1) for i in range(400)]
    idx = P.Places(poi_table(rows))
    for _ in range(40):
        (lat, lon), km = point(), rng.choice([0, 0.2, 1, 2, 5, 50])
        cat = rng.choice([None, "A"])
        brute = {i for i, c, a, b, _ in rows if haversine(lat, lon, a, b) <= km and cat in (None, c)}
        got = idx.nearby(lat, lon, km, cat)
        assert {p.poi_id for p, _ in got} == brute
        assert [d for _, d in got] == sorted(d for _, d in got)


# ---------------------------------------------------------------- demo users
def repeated(user, first_id, here, then, n, minutes=45):
    """n Friday trips here -> then."""
    out = []
    for i in range(n):
        start = pd.Timestamp("2013-01-04 18:00") + pd.Timedelta(weeks=i)
        out.append(trip(first_id + i, user, (here, str(start)), (then, str(start + pd.Timedelta(minutes=minutes)))))
    return out


DEMO_TRIPS = (repeated(1, 0, 1, 2, 6)                    # gym -> food: a persona
              + repeated(2, 20, 4, 2, 9)                 # starts at a home: never shown
              + repeated(3, 40, 1, 2, 9, minutes=1)      # double check-in
              + repeated(4, 60, 1, 2, 4)                 # too few visits
              + repeated(5, 80, 3, 3, 9)                 # same place again
              + [trip(99, 6, (1, "2013-01-01 10:00"), (3, "2013-01-01 11:00"))])  # thin history


def test_only_clear_showable_routines_become_personas(use):
    use(DEMO_TRIPS)
    users = P.demo_users("synth")
    personas = [u for u in users if u["persona"]]
    assert [u["user_id"] for u in personas] == ["1"] and users[0] is personas[0]
    p = personas[0]["persona"]
    assert (p["from_poi"], p["to_poi"], p["count"], p["total"], p["weekday"], p["hour"]) == (1, 2, 6, 6, 4, 18)


def test_demo_users_fields_order_and_n(use):
    use(DEMO_TRIPS)
    users = P.demo_users("synth")
    assert len(users) == 6 and len({u["user_id"] for u in users}) == 6
    assert all({"user_id", "n_checkins", "n_trips", "top_category"} <= u.keys() for u in users)
    assert users[0]["top_category"] == "Burger Joint"  # tied with Gym: alphabetical
    rest = [u["n_checkins"] for u in users[1:]]
    assert rest == sorted(rest, reverse=True)
    assert len(P.demo_users("synth", 2)) == 2


def test_a_tie_is_not_a_clear_routine(use):
    use(repeated(1, 0, 1, 2, 4) + repeated(1, 20, 1, 3, 4))
    assert not any(u["persona"] for u in P.demo_users("synth"))


def test_a_routine_place_must_be_one_of_the_users_top_places(use):
    often = [trip(100 + i, 1, *((p, f"2013-03-{i + 1:02d} {9 + k}:00") for k, p in enumerate((3, 4, 5, 6, 7))))
             for i in range(9)]
    use(repeated(1, 0, 1, 2, 6) + often)
    assert not any(u["persona"]["to_poi"] == 2 for u in P.demo_users("synth") if u["persona"])


def test_at_most_five_personas_each_a_different_user(use):
    use([t for u in range(1, 9) for t in repeated(u, u * 20, 1, 2, 6)])
    personas = [u["user_id"] for u in P.demo_users("synth") if u["persona"]]
    assert len(personas) == len(set(personas)) == 5


def test_returned_users_do_not_alias_the_cache(use):
    use(DEMO_TRIPS)
    P.demo_users("synth")[0]["persona"]["count"] = -1
    assert P.demo_users("synth")[0]["persona"]["count"] == 6


def test_demo_file_has_personas_and_three_scenarios(use, tmp_path):
    use(DEMO_TRIPS)
    out = tmp_path / "assistant" / "demo_users.json"
    P.write_demo_users(["synth"], out)
    d = json.loads(out.read_text(encoding="utf-8"))["datasets"]["synth"]
    assert [u["user_id"] for u in d["personas"]] == ["1"]
    s = d["scenarios"]
    assert (s["habit"]["user_id"], s["habit"]["current_poi"]) == ("1", 1)
    assert s["something_new"]["want_new"] and s["something_new"]["shows"]["unvisited_nearby"] == 3
    assert (s["thin_evidence"]["user_id"], s["thin_evidence"]["current_poi"]) == ("6", 1)
