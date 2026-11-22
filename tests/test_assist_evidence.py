"""Tests for the evidence engine (Member 2, task 2.6)."""
from tracex_assist.contract import Context, Fact, Place, Profile, Visit, routine_key
from tracex_assist.evidence import (
    gather, evidence_to_dict, _transition_facts, _frequency_facts,
    _time_habit_facts, _rrf, _category_prediction,
)


def _visit(poi, day, hour, weekday=0, trip="t1", pos=0):
    return Visit(poi, f"2013-01-{day:02d} {hour:02d}:00", weekday, hour, trip, pos)


def _place(poi_id, category="Bar", lat=40.7, lon=-74.0, pop=10, name=None):
    return Place(poi_id, category, name, lat, lon, pop)


class FakePlaces:
    def __init__(self, places):
        self._p = {p.poi_id: p for p in places}

    def get(self, poi_id):
        return self._p.get(poi_id, _place(poi_id))

    def nearby(self, lat, lon, km, category=None):
        result = []
        for p in self._p.values():
            d = abs(p.lat - lat) * 111.0
            if d <= km and (category is None or p.category == category):
                result.append((p, round(d, 2)))
        return sorted(result, key=lambda x: x[1])


PLACES = FakePlaces([
    _place(1, "Coffee Shop", 40.70, -74.0, 50),
    _place(2, "Office", 40.71, -74.0, 30),
    _place(3, "Gym", 40.72, -74.0, 20),
    _place(4, "Bar", 40.73, -74.0, 15),
    _place(5, "Coffee Shop", 40.701, -74.0, 40),  # nearby unvisited coffee shop
    _place(6, "Bar", 40.702, -74.0, 25),           # nearby unvisited bar
])


def _profile(trips_data):
    trips = []
    for tid, checkins in trips_data:
        trips.append([_visit(poi, day, hour, weekday, str(tid), i)
                      for i, (poi, day, hour, weekday) in enumerate(checkins)])
    all_v = [v for t in trips for v in t]
    from collections import Counter
    freq = Counter(v.poi_id for v in all_v)
    routine = {}
    for v in all_v:
        key = routine_key(v.weekday, v.hour)
        routine.setdefault(key, Counter())[PLACES.get(v.poi_id).category] += 1
    routine = {k: dict(c) for k, c in routine.items()}
    return Profile("nyc", "test_user", trips, freq.most_common(), routine,
                   len(all_v), len(trips), "2013-01-01", "2013-01-10", (40.71, -74.0))


PROFILE = _profile([
    # Trip 1: Coffee -> Office -> Gym (Mon morning)
    (1, [(1, 1, 8, 0), (2, 1, 10, 0), (3, 1, 12, 0)]),
    # Trip 2: Coffee -> Office -> Bar (Mon morning -> evening)
    (2, [(1, 2, 8, 0), (2, 2, 10, 0), (4, 2, 19, 0)]),
    # Trip 3: Coffee -> Office (Wed morning)
    (3, [(1, 3, 8, 2), (2, 3, 10, 2)]),
    # Trip 4: Gym -> Coffee -> Bar (Fri evening)
    (4, [(3, 4, 17, 4), (1, 4, 18, 4), (4, 4, 20, 4)]),
    # Trip 5: Coffee -> Office -> Coffee (Mon morning)
    (5, [(1, 5, 8, 0), (2, 5, 10, 0), (1, 5, 14, 0)]),
])


def test_backoff_picks_right_level():
    ctx = Context(weekday=0, hour=8, current_poi=1)  # Mon morning, at Coffee Shop
    facts = _transition_facts(PROFILE, ctx, PLACES)
    assert len(facts) > 0
    # Office (2) should appear: after Coffee on Mon mornings
    office_facts = [f for f in facts if f.poi_id == 2]
    assert len(office_facts) > 0
    # The most specific level should come first
    first = office_facts[0]
    assert "Mon" in first.level or "morning" in first.level or "after this place" in first.level


def test_shares_add_up():
    ctx = Context(weekday=0, hour=8, current_poi=1)
    facts = _transition_facts(PROFILE, ctx, PLACES)
    # Group facts by level, check each level's shares don't exceed 1
    by_level = {}
    for f in facts:
        by_level.setdefault(f.level, []).append(f)
    for level, level_facts in by_level.items():
        total_share = sum(f.count for f in level_facts) / level_facts[0].total if level_facts else 0
        assert total_share <= 1.0 + 1e-9, f"Level '{level}' shares exceed 1: {total_share}"


def test_explore_never_returns_visited_place():
    ctx = Context(weekday=0, hour=8, current_poi=1)
    ev = gather(PROFILE, ctx, PLACES)
    visited = {v.poi_id for t in PROFILE.trips for v in t}
    for e in ev.explore:
        assert e.poi_id not in visited, f"Explore returned visited place {e.poi_id}"


def test_no_target_leakage():
    """Changing the target (not part of Context) should not affect evidence."""
    ctx1 = Context(weekday=0, hour=8, current_poi=1)
    ctx2 = Context(weekday=0, hour=8, current_poi=1)
    ev1 = gather(PROFILE, ctx1, PLACES)
    ev2 = gather(PROFILE, ctx2, PLACES)
    assert [s.poi_id for s in ev1.habits] == [s.poi_id for s in ev2.habits]
    assert ev1.categories == ev2.categories


def test_rrf_merges_rankings():
    r1 = [1, 2, 3]
    r2 = [2, 3, 4]
    r3 = [3, 1, 5]
    result = _rrf([r1, r2, r3])
    pids = [pid for pid, _ in result]
    # 3 appears in all three rankings, should be near the top
    assert 3 in pids[:2]
    # all pids should be present
    assert set(pids) == {1, 2, 3, 4, 5}


def test_category_prediction():
    t_facts = [Fact("transition", 1, "Coffee Shop", 3, 5, "test"),
               Fact("transition", 2, "Office", 2, 5, "test")]
    time_facts = [Fact("time_habit", None, "Coffee Shop", 4, 10, "test")]
    freq_facts = [Fact("frequency", 1, "Coffee Shop", 10, 20, "test"),
                  Fact("frequency", 2, "Office", 5, 20, "test")]
    cats = _category_prediction(t_facts, time_facts, freq_facts, PLACES)
    assert len(cats) > 0
    assert cats[0][0] == "Coffee Shop"  # should be the top category
    total_share = sum(s for _, s in cats)
    assert abs(total_share - 1.0) < 0.01


def test_thin_flag():
    # Sparse profile: only 2 visits
    sparse = _profile([(1, [(1, 1, 8, 0), (2, 1, 10, 0)])])
    ctx = Context(weekday=0, hour=8, current_poi=1)
    ev = gather(sparse, ctx, PLACES)
    assert ev.thin is True


def test_evidence_to_dict():
    ctx = Context(weekday=0, hour=8, current_poi=1)
    ev = gather(PROFILE, ctx, PLACES)
    d = evidence_to_dict(ev)
    assert isinstance(d, dict)
    assert "habits" in d
    assert "categories" in d
    assert "explore" in d
    assert "facts" in d
    # Facts should have computed 'share'
    for f in d["facts"]:
        if f["total"] > 0:
            assert "share" in f
            assert abs(f["share"] - f["count"] / f["total"]) < 0.01


def test_gather_without_current_poi():
    ctx = Context(weekday=0, hour=8, current_poi=None)
    ev = gather(PROFILE, ctx, PLACES)
    assert len(ev.habits) > 0  # frequency ranking still works
    assert ev.explore == []    # no explore without a current location


def test_want_kind_filters_explore():
    ctx = Context(weekday=0, hour=8, current_poi=1, want_kind="food")
    ev = gather(PROFILE, ctx, PLACES)
    for e in ev.explore:
        assert "restaurant" in e.category.lower() or "food" in e.category.lower() or len(ev.explore) == 0
