"""Evidence engine for the personal assistant (Member 2, tasks 2.1-2.4).

    from tracex_assist.evidence import gather, evidence_to_dict
    ev = gather(profile, context, places)

Back-off transition statistics -> Markov, frequency, time-habit rankings -> reciprocal rank fusion ->
category prediction -> explore list of unvisited nearby places. Every suggestion keeps its supporting
facts with counts and the back-off level used.

Leakage rule: `gather` reads context.weekday, context.hour and context.current_poi (the last prefix
place) but never the target place. `evidence_to_dict` is JSON-safe for the LLM prompt and the UI.
"""
from __future__ import annotations

from collections import Counter

from .contract import (
    Context, Evidence, Explore, Fact, Place, Profile, Suggestion,
    WEEKDAYS, category_matches, routine_key, time_bucket, to_dict,
)

RRF_K = 60
TOP_HABITS = 5
TOP_CATEGORIES = 3
TOP_EXPLORE = 5
THIN_THRESHOLD = 5
EXPLORE_RADII = (1.0, 2.0, 5.0)


def _all_visits(profile: Profile):
    return [v for t in profile.trips for v in t]


# ---------------------------------------------------------------------------
# 2.1  Transition statistics with back-off
# ---------------------------------------------------------------------------
def _transition_facts(profile: Profile, context: Context, places) -> list[Fact]:
    """After-this-place counts, backed off from narrow to broad."""
    if context.current_poi is None:
        return []

    visits = _all_visits(profile)
    bucket = time_bucket(context.hour)
    weekday_name = WEEKDAYS[context.weekday]

    pairs_specific: Counter = Counter()   # same place, same weekday, same time bucket
    pairs_time: Counter = Counter()       # same place, same time bucket
    pairs_place: Counter = Counter()      # same place, any time
    pairs_category: Counter = Counter()   # same category, any time

    current_cat = places.get(context.current_poi).category

    for trip in profile.trips:
        for a, b in zip(trip, trip[1:]):
            if a.poi_id == context.current_poi:
                pairs_place[b.poi_id] += 1
                if time_bucket(a.hour) == bucket:
                    pairs_time[b.poi_id] += 1
                    if a.weekday == context.weekday:
                        pairs_specific[b.poi_id] += 1
            if places.get(a.poi_id).category == current_cat:
                pairs_category[b.poi_id] += 1

    n_specific = sum(pairs_specific.values())
    n_time = sum(pairs_time.values())
    n_place = sum(pairs_place.values())
    n_category = sum(pairs_category.values())

    facts = []
    seen = set()

    levels = [
        (pairs_specific, n_specific, f"you, after this place, on {weekday_name} {bucket}s"),
        (pairs_time, n_time, f"you, after this place, {bucket}s"),
        (pairs_place, n_place, "you, after this place, any day"),
        (pairs_category, n_category, f"everyone who visited a {current_cat}, any day"),
    ]

    for counts, total, level in levels:
        if total < 1:
            continue
        for pid, c in counts.most_common():
            if pid not in seen:
                seen.add(pid)
                cat = places.get(pid).category
                facts.append(Fact("transition", pid, cat, c, total, level))

    return facts


# ---------------------------------------------------------------------------
# 2.1b  Frequency and time-habit facts
# ---------------------------------------------------------------------------
def _frequency_facts(profile: Profile, places) -> list[Fact]:
    visits = _all_visits(profile)
    freq = Counter(v.poi_id for v in visits)
    total = len(visits)
    return [Fact("frequency", pid, places.get(pid).category, c, total, "you, all visits")
            for pid, c in freq.most_common()]


def _time_habit_facts(profile: Profile, context: Context, places) -> list[Fact]:
    key = routine_key(context.weekday, context.hour)
    bucket_data = profile.routine.get(key, {})
    total = sum(bucket_data.values())
    if total == 0:
        return []
    weekday_name = WEEKDAYS[context.weekday]
    bucket = time_bucket(context.hour)
    return [Fact("time_habit", None, cat, count, total,
                 f"you, on {weekday_name} {bucket}s")
            for cat, count in Counter(bucket_data).most_common()]


# ---------------------------------------------------------------------------
# 2.2  Ranking and reciprocal rank fusion
# ---------------------------------------------------------------------------
def _rank_list(facts: list[Fact], key="poi_id") -> list[int]:
    seen = set()
    result = []
    for f in facts:
        pid = getattr(f, key) if key == "poi_id" else f.poi_id
        if pid is not None and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def _rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _markov_ranking(profile: Profile, context: Context) -> list[int]:
    if context.current_poi is None:
        return []
    after: Counter = Counter()
    for trip in profile.trips:
        for a, b in zip(trip, trip[1:]):
            if a.poi_id == context.current_poi:
                after[b.poi_id] += 1
    return [pid for pid, _ in after.most_common()]


def _frequency_ranking(profile: Profile) -> list[int]:
    visits = _all_visits(profile)
    return [pid for pid, _ in Counter(v.poi_id for v in visits).most_common()]


def _time_habit_ranking(profile: Profile, context: Context, places) -> list[int]:
    """Rank POIs by how well their category matches the time-of-day pattern."""
    key = routine_key(context.weekday, context.hour)
    bucket_data = profile.routine.get(key, {})
    if not bucket_data:
        return []
    cat_rank = {cat: rank for rank, (cat, _) in enumerate(Counter(bucket_data).most_common())}
    visits = _all_visits(profile)
    freq = Counter(v.poi_id for v in visits)
    poi_cats = {pid: places.get(pid).category for pid in freq}
    scored = []
    for pid, count in freq.items():
        cat = poi_cats[pid]
        if cat in cat_rank:
            scored.append((pid, cat_rank[cat], -count))
    scored.sort(key=lambda x: (x[1], x[2]))
    return [pid for pid, _, _ in scored]


# ---------------------------------------------------------------------------
# 2.3  Category prediction and explore list
# ---------------------------------------------------------------------------
def _category_prediction(transition_facts: list[Fact], time_facts: list[Fact],
                         freq_facts: list[Fact], places) -> list[tuple[str, float]]:
    cat_scores: Counter = Counter()
    for f in transition_facts:
        if f.poi_id is not None:
            cat_scores[places.get(f.poi_id).category] += f.share
    for f in time_facts:
        if f.category:
            cat_scores[f.category] += f.share
    for f in freq_facts[:10]:
        if f.poi_id is not None:
            cat_scores[places.get(f.poi_id).category] += f.share * 0.3
    total = sum(cat_scores.values())
    if total == 0:
        return []
    return [(cat, round(s / total, 3)) for cat, s in cat_scores.most_common(TOP_CATEGORIES)]


def _explore_list(context: Context, profile: Profile, places,
                  categories: list[tuple[str, float]]) -> list[Explore]:
    if context.current_poi is None:
        return []
    here = places.get(context.current_poi)
    visited = {v.poi_id for v in _all_visits(profile)}
    target_cat = None
    if context.want_kind is None and categories:
        target_cat = categories[0][0]

    for radius in EXPLORE_RADII:
        candidates = []
        for p, km in places.nearby(here.lat, here.lon, radius, target_cat):
            if p.poi_id in visited:
                continue
            if not category_matches(context.want_kind, p.category):
                continue
            candidates.append(Explore(p.poi_id, round(km, 2), p.popularity, p.category))
        if candidates:
            candidates.sort(key=lambda e: -e.popularity)
            return candidates[:TOP_EXPLORE]
    return []


# ---------------------------------------------------------------------------
# 2.4  Confidence and thin flag
# ---------------------------------------------------------------------------
_OFFLINE_CACHE: dict[str, dict] = {}


def _offline_confidence(dataset: str) -> dict:
    if dataset not in _OFFLINE_CACHE:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "reports" / "assistant" / "results.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for d, row in data.get("engine", {}).items():
                _OFFLINE_CACHE[d] = {"acc1": row.get("acc1"), "acc5": row.get("acc5"),
                                     "n": row.get("n", 0)}
        else:
            _OFFLINE_CACHE.setdefault(dataset, {"acc1": None, "acc5": None, "n": 0})
    return _OFFLINE_CACHE.get(dataset, {"acc1": None, "acc5": None, "n": 0})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def gather(profile: Profile, context: Context, places) -> Evidence:
    """Build evidence for the user's next place. No target leakage."""
    transition_facts = _transition_facts(profile, context, places)
    freq_facts = _frequency_facts(profile, places)
    time_facts = _time_habit_facts(profile, context, places)

    # Build separate rankings and fuse
    markov = _markov_ranking(profile, context)
    frequency = _frequency_ranking(profile)
    time_habit = _time_habit_ranking(profile, context, places)

    rankings = [r for r in [markov, frequency, time_habit] if r]
    fused = _rrf(rankings)

    # Collect facts per suggestion
    fact_index: dict[int, list[Fact]] = {}
    for f in transition_facts + freq_facts:
        if f.poi_id is not None:
            fact_index.setdefault(f.poi_id, []).append(f)

    habits = []
    for pid, score in fused[:TOP_HABITS]:
        facts = fact_index.get(pid, [])
        if not facts:
            cat = places.get(pid).category
            visits = _all_visits(profile)
            cnt = sum(1 for v in visits if v.poi_id == pid)
            facts = [Fact("frequency", pid, cat, cnt, len(visits), "you, all visits")]
        habits.append(Suggestion(pid, round(score, 4), facts))

    categories = _category_prediction(transition_facts, time_facts, freq_facts, places)
    explore = _explore_list(context, profile, places, categories)

    all_facts = transition_facts + freq_facts + time_facts
    visits = _all_visits(profile)
    n_transition = sum(1 for t in profile.trips for a, b in zip(t, t[1:])
                       if context.current_poi is not None and a.poi_id == context.current_poi)
    thin = (n_transition if context.current_poi is not None else len(visits)) < THIN_THRESHOLD

    confidence = _offline_confidence(profile.dataset)

    return Evidence(context, habits, categories, explore, all_facts, confidence, thin)


def evidence_to_dict(ev: Evidence) -> dict:
    return to_dict(ev)
