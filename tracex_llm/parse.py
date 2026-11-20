"""Parse and validate LLM answers into ranked POI lists (Member 3).

    res = parse_response(text, valid_ids=set(ds.pois.poi_id),
                         input_ids=user_poi_ids(inst, history),
                         prompt_ids=set(prompt.meta["prompt_poi_ids"]))
    res.ranked         # de-duplicated POI ids in the model's order (scored as is)
    res.to_dict()      # flat record stored in the results JSONL

Policy (stated in the paper):
- Invalid ids (not a POI of the dataset) stay in `ranked` at the position the model gave
  them. They never match the target, and removing them would move later valid ids up and
  inflate MRR/NDCG.
- Failed or short answers are NOT padded with heuristic guesses; a miss stays a miss and
  shows up in the invalid-answer rate.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field

from . import config as C

_INT_ITEM = re.compile(r"^\s*(?:POI|poi|id|ID)?\s*[#:]?\s*(-?\d+)\s*$")
_LIST_AFTER_KEY = re.compile(r"[\"']?prediction[s]?[\"']?\s*[:=]\s*\[([^\]]*)\]", re.IGNORECASE)
_POI_MENTION = re.compile(r"\bPOI\s*(?:ID)?\s*[#:]?\s*(\d+)", re.IGNORECASE)
_REASON = re.compile(r"[\"']reason[\"']\s*:\s*[\"'](.*?)[\"']\s*[,}]", re.IGNORECASE | re.DOTALL)


@dataclass
class ParseResult:
    status: str                    # json | literal | regex_list | poi_mentions | empty | failed
    ranked: list = field(default_factory=list)   # de-duplicated ids, model order, capped at N
    reason: str | None = None
    n_returned: int = 0            # ids the model returned (before de-duplication / cap)
    n_duplicates: int = 0
    invalid_ids: list = field(default_factory=list)
    top1: int | None = None
    top1_valid: bool = False
    invalid_answer: bool = True    # brief's "invalid POI" rate: no usable top-1 prediction
    copied_top1: bool | None = None         # top-1 is a POI from the user's input (history + prefix)
    copied_top1_prompt: bool | None = None  # top-1 is a POI shown in the prompt
    copied_top5_frac: float | None = None   # share of the top-5 taken from the user's input

    def to_dict(self) -> dict:
        return asdict(self)


def user_poi_ids(inst: dict, history: list[dict]) -> set[int]:
    """POIs in the model's possible input: the full history plus the current prefix.

    Same definition as the data pipeline's `is_revisit`, built only from inputs (never the target).
    """
    ids = {c["poi_id"] for t in history for c in t["checkins"]}
    ids.update(c["poi_id"] for c in inst["prefix"])
    return ids


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _candidate_objects(text: str):
    """Balanced {...} substrings, outermost first."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _to_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        m = _INT_ITEM.match(x)
        return int(m.group(1)) if m else None
    if isinstance(x, dict):  # e.g. {"poi_id": 12, ...}
        for k in ("poi_id", "poi", "id"):
            if k in x:
                return _to_int(x[k])
    return None


def _from_object(obj):
    if not isinstance(obj, dict):
        return None, None
    key = next((k for k in obj if str(k).lower() in ("prediction", "predictions")), None)
    if key is None or not isinstance(obj[key], list):
        return None, None
    reason = obj.get("reason")
    return obj[key], reason if isinstance(reason, str) else None


def extract(text: str):
    """Return (status, raw items, reason)."""
    body = _strip_fences(text or "")
    for cand in _candidate_objects(body):
        for status, loader in (("json", json.loads), ("literal", ast.literal_eval)):
            try:
                items, reason = _from_object(loader(cand))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if items is not None:
                return status, items, reason
    m = _REASON.search(body)
    reason = m.group(1) if m else None
    m = _LIST_AFTER_KEY.search(body)
    if m:
        items = [s.strip().strip("\"'") for s in m.group(1).split(",") if s.strip()]
        return "regex_list", items, reason
    mentions = _POI_MENTION.findall(body)
    if mentions:
        return "poi_mentions", mentions, reason
    return "failed", [], reason


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def parse_response(text: str, valid_ids: set[int] | None = None, input_ids: set[int] | None = None,
                   prompt_ids: set[int] | None = None, n: int = C.N_PREDICTIONS) -> ParseResult:
    status, items, reason = extract(text)
    ids = [i for i in (_to_int(x) for x in items) if i is not None]
    ranked, seen = [], set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            ranked.append(i)
    n_dup = len(ids) - len(ranked)
    ranked = ranked[:n]
    if not ranked and status != "failed":
        status = "empty"

    res = ParseResult(status=status, ranked=ranked, reason=reason,
                      n_returned=len(items), n_duplicates=n_dup)
    if valid_ids is not None:
        res.invalid_ids = [i for i in ranked if i not in valid_ids]
    if ranked:
        res.top1 = ranked[0]
        res.top1_valid = valid_ids is None or ranked[0] in valid_ids
    res.invalid_answer = not res.top1_valid
    if ranked and input_ids is not None:
        res.copied_top1 = ranked[0] in input_ids
        top5 = ranked[:5]
        res.copied_top5_frac = sum(i in input_ids for i in top5) / len(top5)
    if ranked and prompt_ids is not None:
        res.copied_top1_prompt = ranked[0] in prompt_ids
    return res
