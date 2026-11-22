"""Shared contract of the personal assistant (docs/assistant_plan.md, section 3).

Change only by agreement of all three members: profile.py builds these objects, evidence.py fills
`Evidence`, llm.py and the dashboard read them.

Cross-member functions (implemented in the owner's module):
    Member 1  profile.place_index(dataset) -> PlaceIndex   (.get(poi_id) -> Place, .nearby(lat, lon, km, category=None))
    Member 1  profile.build_profile(dataset, user_id, history=None) -> Profile
    Member 1  profile.demo_users(dataset, n=20) -> list[dict]   (user_id, n_checkins, n_trips, top_category)
    Member 2  evidence.gather(profile, context, places) -> Evidence
    Member 2  evidence.evidence_to_dict(ev) -> dict             (JSON-safe, for the LLM prompt and the UI)
    Member 3  llm.answer(question, ev, model=...) -> Iterator[str], llm.fallback_answer(ev), llm.ollama_status()

Leakage rule (for the offline evaluation): with `history=ds.history(inst)` a profile uses only those trips, and
`gather` may use the context (target weekday/hour, last prefix place) but never the target place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Hour ranges [start, end); "night" wraps past midnight.
TIME_BUCKETS = {"morning": (5, 11), "midday": (11, 14), "afternoon": (14, 17), "evening": (17, 21), "night": (21, 5)}


def time_bucket(hour: int) -> str:
    for name, (a, b) in TIME_BUCKETS.items():
        if (a <= hour < b) if a < b else (hour >= a or hour < b):
            return name
    raise ValueError(f"hour out of range: {hour}")


def routine_key(weekday: int, hour: int) -> str:
    """Key of Profile.routine, e.g. 'Fri|evening'."""
    return f"{WEEKDAYS[weekday]}|{time_bucket(hour)}"


@dataclass
class Place:
    """One POI."""
    poi_id: int
    category: str
    name: str | None  # None on NYC (no venue names)
    lat: float
    lon: float
    popularity: int  # check-ins by all users in train


@dataclass
class Visit:
    """One check-in of the user."""
    poi_id: int
    time: str  # "2013-01-04 18:05"
    weekday: int  # 0 = Monday
    hour: int
    trip: str  # trajectory id
    pos: int  # position inside the trip


@dataclass
class Profile:
    """A user's history, built by Member 1."""
    dataset: str
    user_id: str
    trips: list[list[Visit]]  # oldest first
    top_places: list[tuple[int, int]]  # (poi_id, visits), most visited first
    routine: dict[str, dict[str, int]]  # routine_key -> {category: visits}
    n_checkins: int
    n_trips: int
    first_day: str
    last_day: str
    home: tuple[float, float]  # centroid (lat, lon) of the user's places


class PlaceIndex(Protocol):
    def get(self, poi_id: int) -> Place: ...

    def nearby(self, lat: float, lon: float, km: float, category: str | None = None) -> list[tuple[Place, float]]: ...


@dataclass
class Context:
    """The situation of the question (from the UI pickers, or from a test instance)."""
    weekday: int
    hour: int
    current_poi: int | None  # where the user is now
    want_new: bool = False  # "suggest something new"


@dataclass
class Fact:
    """One piece of evidence, always with its counts."""
    kind: str  # "transition" | "time_habit" | "frequency" | "similar_trip" | "category"
    poi_id: int | None
    category: str | None
    count: int  # e.g. 11
    total: int  # e.g. 20 -> share 0.55
    level: str  # back-off level in words, e.g. "you, after this place, on Fridays"

    @property
    def share(self) -> float:
        return self.count / self.total if self.total else 0.0


@dataclass
class Suggestion:
    poi_id: int
    score: float
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Explore:
    """A place the user has never visited."""
    poi_id: int
    km: float
    popularity: int
    category: str


@dataclass
class Evidence:
    """Returned by Member 2's gather(); explained by llm.py and shown by the dashboard."""
    context: Context
    habits: list[Suggestion]  # top 5 known places
    categories: list[tuple[str, float]]  # top 3 next categories with share
    explore: list[Explore]  # up to 5 new places
    facts: list[Fact]  # everything shown in "How I got this"
    confidence: dict  # {"acc1": 0.25, "acc5": 0.59, "n": 500} from the offline evaluation
    thin: bool  # True if the evidence rests on fewer than 5 visits


def to_dict(obj) -> dict:
    """Plain-dict view of any contract dataclass (adds Fact.share)."""
    d = asdict(obj)

    def add_share(x):
        if isinstance(x, dict):
            if {"count", "total", "kind"} <= x.keys():
                x["share"] = round(x["count"] / x["total"], 3) if x["total"] else 0.0
            for v in x.values():
                add_share(v)
        elif isinstance(x, list):
            for v in x:
                add_share(v)
    add_share(d)
    return d
