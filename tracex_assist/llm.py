"""Grounded explanation layer of the assistant (Member 3, tasks 3.2-3.4).

The evidence engine decides; the LLM only words the evidence. Flow:
    view = evidence_view(ev, places)        # readable labels, every number pre-formatted as text
    res = answer(question, ev, places)       # Ollama -> validate -> one stricter retry -> template fallback
Every place ID and every number in an accepted answer must appear in the evidence (`validate`). If Ollama is not
running, or the model keeps inventing things, the template answer built from the same evidence is returned.
Uses only the standard library (Ollama's HTTP API on localhost); no API key, nothing leaves the laptop.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .contract import WEEKDAYS, Evidence, time_bucket

OLLAMA_URL = os.environ.get("TRACEX_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = "qwen2.5:3b"
MODELS = ("qwen2.5:3b", "qwen2.5:7b")
OPTIONS = {"temperature": 0.2, "num_predict": 320}
TIMEOUT_S = 90
NEW_WORDS = re.compile(r"\b(new|different|else|other|try|tried|trying|explore|discover|never been|"
                       r"haven'?t (been|tried|visited)|not (been|tried|visited))\b", re.I)
# Question words -> contract.KINDS key; checked in this order ("coffee shop" is coffee, not shopping).
KIND_WORDS = [
    ("coffee", r"coffee|caf[eé]|\btea\b|latte|espresso"),
    ("drinks", r"\bdrinks?\b|\bbar\b|beer|pub|wine|cocktail|nightlife|club"),
    ("food", r"\beat\b|food|hungry|lunch|dinner|breakfast|brunch|restaurant|snack|meal|burger|pizza|sushi|ramen"),
    ("shopping", r"\bshop(ping)?\b|\bbuy\b|store|mall|groceries|market"),
    ("outdoors", r"\bpark\b|\bwalk\b|outside|outdoors|nature|fresh air|garden|beach"),
    ("transport", r"station|\btrain\b|\bbus\b|subway|commute|airport"),
]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _get(path: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_status(model: str = DEFAULT_MODEL) -> dict:
    """{'running', 'version', 'models', 'has_model', 'error'}; never raises."""
    try:
        version = _get("/api/version").get("version")
        models = sorted(m["name"] for m in _get("/api/tags").get("models", []))
        return {"running": True, "version": version, "models": models, "has_model": model in models, "error": None}
    except Exception as e:  # not installed, not started, or blocked
        return {"running": False, "version": None, "models": [], "has_model": False, "error": str(e)}


def chat(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False, "options": OPTIONS}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Evidence -> readable view (the only thing the model sees)
# ---------------------------------------------------------------------------
def label(places, poi_id: int) -> str:
    p = places.get(poi_id)
    return f"{p.name} ({p.category} #{p.poi_id})" if p.name else f"{p.category} #{p.poi_id}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _why(f) -> str:
    """One fact as a natural phrase; the numbers are what the validator will accept."""
    n = f"{f.count} of {f.total}"
    p = _pct(f.share)
    phrase = {
        "transition": f"you went there next {n} times ({p})",
        "frequency": f"{n} of your check-ins were there ({p})",
        "time_habit": f"{n} of your visits at this time ({p})",
        "similar_trip": f"it came next in {n} similar past trips ({p})",
        "category": f"{n} of your next stops were this kind of place ({p})",
    }.get(f.kind, f"{n} ({p})")
    return f"{phrase} [based on: {f.level}]" if f.level else phrase


def wants_new(question: str) -> bool:
    return bool(NEW_WORDS.search(question or ""))


def wanted_kind(question: str) -> str | None:
    for kind, pat in KIND_WORDS:
        if re.search(pat, question or "", re.I):
            return kind
    return None


def parse_intent(question: str) -> dict:
    """Call BEFORE gather(): copy the result into Context(want_new=..., want_kind=...)."""
    return {"want_new": wants_new(question), "want_kind": wanted_kind(question)}


def evidence_view(ev: Evidence, places) -> dict:
    c = ev.context
    where = f", currently at {label(places, c.current_poi)}" if c.current_poi is not None else ""
    view = {
        "situation": f"{WEEKDAYS[c.weekday]} {c.hour:02d}:00 ({time_bucket(c.hour)}){where}",
        "likely_next": [{"place": label(places, s.poi_id), "why": [_why(f) for f in s.facts]} for s in ev.habits],
        "likely_kind_of_place": [f"{cat}: {_pct(share)}" for cat, share in ev.categories],
        # new places only when asked for (or nothing else to offer): keeps answers focused on the question
        "new_places": [{"place": label(places, e.poi_id), "distance": f"{e.km:.1f} km",
                        "popularity": f"{e.popularity} check-ins by other people"} for e in ev.explore]
        if (c.want_new or not ev.habits) else [],
        "evidence_strength": "limited (few past visits)" if ev.thin else "good",
        "wants_something_new": bool(c.want_new),
    }
    if c.want_kind:
        view["asked_for"] = c.want_kind
    acc = ev.confidence.get("acc1") if ev.confidence else None
    if acc:
        view["accuracy"] = (f"in an offline test the top suggestion was the real next place {_pct(acc)} of the time"
                            f" (n = {ev.confidence.get('n')})")
    return view


SYSTEM = """You are TraceX, a friendly personal assistant that knows the user's past check-ins.
You receive EVIDENCE computed from the user's history. Answer the user's question using ONLY that evidence.

Rules:
1. Mention places exactly as written in the evidence, including the # number. Never invent a place.
2. Copy numbers exactly as written (for example "11 of 20 times (55%)"). Never calculate or round new numbers.
3. Start with the most likely next place and say why, using its "why" lines.
4. If the user wants something new, or "wants_something_new" is true, suggest places from "new_places" only,
   with their distance. If "new_places" is empty, say there are no new places nearby in the data.
5. Only if "evidence_strength" is limited, say that the suggestion rests on few past visits.
6. If "accuracy" is given, end with one short sentence about it.
7. Do not add any fact that is not in the evidence: no walking times, opening hours, descriptions or opinions
   about the places.
8. Places in "likely_next" are places the user already knows. Never call them new.
9. If "asked_for" is given (for example "food"), focus on that kind of place.
10. Speak to the user as "you". At most 120 words. No lists longer than 3 items. No JSON.
11. Stop after the facts: no closing remarks, no "you will enjoy it", no sales talk."""


def build_messages(question: str, view: dict, stricter: list[str] | None = None) -> list[dict]:
    user = f"EVIDENCE:\n{json.dumps(view, ensure_ascii=False, indent=1)}\n\nQUESTION: {question.strip() or 'Where should I go next?'}"
    if stricter:
        user += ("\n\nYour previous answer broke the rules: " + "; ".join(stricter) +
                 ". Use only places and numbers that appear in the EVIDENCE, written exactly as there.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Validation: every place ID and number in the answer must come from the evidence
# ---------------------------------------------------------------------------
_ID = re.compile(r"#\s?(\d+)")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_OF = re.compile(r"\b(\d+)\s+(?:times?\s+|visits?\s+)?(?:out\s+)?of\s+(?:your\s+|the\s+)?(?:last\s+)?(\d+)\b", re.I)
_KM = re.compile(r"(\d+(?:\.\d+)?)\s?km\b", re.I)


def validate(text: str, view: dict) -> list[str]:
    src = json.dumps(view, ensure_ascii=False)
    ids = set(_ID.findall(src))
    pcts = {float(x) for x in _PCT.findall(src)}
    pairs = set(_OF.findall(src))
    kms = {float(x) for x in _KM.findall(src)}
    problems = []
    bad_ids = sorted({i for i in _ID.findall(text) if i not in ids}, key=int)
    if bad_ids:
        problems.append("places not in the evidence: " + ", ".join("#" + i for i in bad_ids))
    bad_p = sorted({x for x in _PCT.findall(text) if float(x) not in pcts}, key=float)
    if bad_p:
        problems.append("percentages not in the evidence: " + ", ".join(x + "%" for x in bad_p))
    bad_of = sorted({f"{a} of {b}" for a, b in _OF.findall(text) if (a, b) not in pairs})
    if bad_of:
        problems.append("counts not in the evidence: " + ", ".join(bad_of))
    bad_km = sorted({x for x in _KM.findall(text) if float(x) not in kms}, key=float)
    if bad_km:
        problems.append("distances not in the evidence: " + ", ".join(x + " km" for x in bad_km))
    # a distance must sit in the same sentence as the new place it belongs to (named by its #id or its name)
    owners = []  # (id, name or None, km)
    for p in view.get("new_places", []):
        name = re.sub(r"\s*\([^()]*#\d+\)$", "", p["place"])
        owners.append((_ID.search(p["place"]).group(1), name if name != p["place"] else None,
                       float(p["distance"].split()[0])))
    for sent in re.split(r"(?<=[.!?。])\s+|\n+", text):
        ids_here = set(_ID.findall(sent))
        for x in _KM.findall(sent):
            if float(x) in kms and not any(km == float(x) and (i in ids_here or (n and n in sent))
                                           for i, n, km in owners):
                problems.append(f"distance {x} km is not next to the place it belongs to")
    if not text.strip():
        problems.append("empty answer")
    return problems


# ---------------------------------------------------------------------------
# Template answer (Ollama off, or the model keeps breaking the rules)
# ---------------------------------------------------------------------------
def fallback_answer(ev: Evidence, places, question: str = "") -> str:
    v = evidence_view(ev, places)
    want_new = v["wants_something_new"] or wants_new(question)
    out = [f"It is {v['situation']}."]
    if v["likely_next"]:
        top = v["likely_next"][0]
        out.append(f"Most likely next: **{top['place']}**" + (f" ({top['why'][0]})." if top["why"] else "."))
        others = [x["place"] for x in v["likely_next"][1:3]]
        if others:
            out.append("Also likely: " + " and ".join(others) + ".")
    elif v["likely_kind_of_place"]:
        out.append("At this time you usually go to: " + ", ".join(v["likely_kind_of_place"]) + ".")
    else:
        out.append("There are no past visits to base a suggestion on.")
    if v["new_places"] and (want_new or not v["likely_next"]):
        out.append("Something new nearby: " + "; ".join(f"{p['place']} ({p['distance']})" for p in v["new_places"][:3])
                   + ".")
    elif want_new:
        out.append("There are no new places of that kind nearby in the data.")
    if ev.thin:
        out.append("This rests on only a few past visits.")
    if v.get("accuracy"):
        out.append(v["accuracy"][0].upper() + v["accuracy"][1:] + ".")
    return " ".join(out)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    text: str
    source: str  # "llm" | "llm-retry" | "template"
    model: str | None
    seconds: float
    problems: list[str] = field(default_factory=list)  # problems of rejected drafts
    view: dict = field(default_factory=dict)  # exactly what the model was shown


def answer(question: str, ev: Evidence, places, model: str = DEFAULT_MODEL, use_llm: bool = True) -> Answer:
    t0 = time.perf_counter()
    if wants_new(question):  # normally already set via parse_intent() before gather()
        ev.context.want_new = True
    view = evidence_view(ev, places)
    problems: list[str] = []
    if use_llm:
        stricter = None
        for attempt in (1, 2):
            try:
                text = chat(build_messages(question, view, stricter), model)
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                problems.append(f"Ollama unavailable: {e}")
                break
            found = validate(text, view)
            if not found:
                return Answer(text, "llm" if attempt == 1 else "llm-retry", model, time.perf_counter() - t0,
                              problems, view)
            problems += found
            stricter = found
    return Answer(fallback_answer(ev, places, question), "template", None, time.perf_counter() - t0, problems, view)
