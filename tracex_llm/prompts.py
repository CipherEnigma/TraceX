"""Prompt builders for the two LLM methods (Member 3).

    from tracex_llm.prompts import build_prompt
    p = build_prompt("llmmob", inst, ds.history(inst))
    p = build_prompt("retrieval", inst, ds.history(inst), idf=idf, retriever="jaccard")
    p.messages()   # chat messages for the backend
    p.meta         # what went into the prompt (for analysis and the dashboard)

Leakage rule: the only target fields ever read are `time` and `weekday`, through
`target_time()`, and only when `config.INCLUDE_TARGET_TIME` is on. The target POI,
category and name never reach a prompt.

Venue names (E3): every builder takes `name_fn(checkin) -> str | None`.
None (the default) shows no names, as for NYC/TKY and the "removed" condition.
`present_names` shows the real Massive-STEPS names; the "aliased" condition passes a
function from `tracex_llm.aliasing`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tracex_ir.retrieval import retrieve_demonstrations

from . import config as C

NameFn = Callable[[dict], "str | None"]
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SYSTEM_PROMPT = (
    "You predict the next place (point of interest, POI) a user will check in at, "
    "using only the user's own check-in records. Every POI is identified by an integer ID. "
    "Reply with a single JSON object and nothing else."
)


def present_names(checkin: dict) -> str | None:
    """E3 'present' condition: the real venue name stored on the check-in."""
    return checkin.get("name")


@dataclass
class Prompt:
    system: str
    user: str
    meta: dict = field(default_factory=dict)

    def messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.user}]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_checkin(c: dict, name_fn: NameFn | None = None) -> str:
    """`2013-01-02 12:39 Wednesday | POI 481 | Department Store | "name"`."""
    parts = [f"{c['time'][:16]} {WEEKDAYS[c['weekday']]}", f"POI {c['poi_id']}", c["category"]]
    name = name_fn(c) if name_fn else None
    if name:
        parts.append(f'"{name}"')
    return " | ".join(parts)


def target_time(inst: dict) -> dict | None:
    """The only access to the target: its time and weekday, nothing else."""
    if not C.INCLUDE_TARGET_TIME:
        return None
    t = inst["target"]
    return {"time": t["time"], "weekday": t["weekday"]}


def _lines(checkins, name_fn, indent="  "):
    return "\n".join(indent + format_checkin(c, name_fn) for c in checkins)


def _format_line(name_fn) -> str:
    fmt = "time weekday | POI id | category"
    return fmt + (' | "venue name"' if name_fn else "")


def _question(inst) -> str:
    tt = target_time(inst)
    when = (f"The user's next check-in is at {tt['time'][:16]} on {WEEKDAYS[tt['weekday']]}. "
            if tt else "")
    return (
        f"{when}Which POI will the user visit next?\n\n"
        f"Return exactly this JSON: "
        f'{{"prediction": [{C.N_PREDICTIONS} distinct integer POI IDs, most likely first], '
        f'"reason": "one short sentence"}}'
    )


def _prefix_block(inst, name_fn):
    prefix = inst["prefix"][-C.PREFIX_MAX_CHECKINS:]
    block = ("<context>\nThe user's current trajectory so far, oldest first:\n"
             f"{_lines(prefix, name_fn)}\n</context>")
    return block, prefix


def _meta(method, inst, shown_checkins, name_fn, **extra):
    return {
        "method": method,
        "instance_id": inst["instance_id"],
        "names": name_fn is not None,
        "target_time_shown": C.INCLUDE_TARGET_TIME,
        "prefix_shown": min(len(inst["prefix"]), C.PREFIX_MAX_CHECKINS),
        "prefix_truncated": len(inst["prefix"]) > C.PREFIX_MAX_CHECKINS,
        # every POI id visible to the model: used for copy-rate and target-in-prompt analysis
        "prompt_poi_ids": sorted({c["poi_id"] for c in shown_checkins}),
        **extra,
    }


# ---------------------------------------------------------------------------
# Method 1: LLM-Mob style (most recent history)
# ---------------------------------------------------------------------------
def recent_history(history: list[dict], max_checkins: int = C.LLMMOB_MAX_HISTORY):
    """Most recent `max_checkins` history check-ins, still grouped by trajectory, oldest first."""
    picked, budget = [], max_checkins
    for traj in reversed(history):
        if budget <= 0:
            break
        ch = traj["checkins"][-budget:]
        picked.append((traj, ch))
        budget -= len(ch)
    return list(reversed(picked))


def build_llmmob(inst: dict, history: list[dict], name_fn: NameFn | None = None) -> Prompt:
    groups = recent_history(history)
    hist_text = "\n".join(
        f"Trajectory {i}:\n{_lines(ch, name_fn)}" for i, (_, ch) in enumerate(groups, 1))
    prefix_block, prefix = _prefix_block(inst, name_fn)
    hist_checkins = [c for _, ch in groups for c in ch]
    user = (
        f"Each check-in is written as: {_format_line(name_fn)}.\n\n"
        "<history>\nThe user's most recent earlier trajectories, oldest first. "
        "They show the user's long-term habits.\n"
        f"{hist_text}\n</history>\n\n"
        f"{prefix_block}\n\n"
        "Use the history for the user's habits, the context for what they are doing now, "
        "and the time of the next check-in.\n"
        f"{_question(inst)}"
    )
    meta = _meta("llmmob", inst, hist_checkins + prefix, name_fn,
                 history_checkins_shown=len(hist_checkins),
                 history_trajs_shown=[t["traj_id"] for t, _ in groups])
    return Prompt(SYSTEM_PROMPT, user, meta)


# ---------------------------------------------------------------------------
# Method 2: retrieved demonstrations (Jaccard / LCS selection, Nishida et al.)
# ---------------------------------------------------------------------------
def build_retrieval(inst: dict, history: list[dict], idf=None, retriever: str = C.DEFAULT_RETRIEVER,
                    top_k: int = C.DEMO_TOP_K, name_fn: NameFn | None = None) -> Prompt:
    if retriever not in C.RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}; expected one of {C.RETRIEVERS}")
    if retriever == "bm25" and idf is None:
        raise ValueError("bm25 retrieval needs an IDF built from the training trajectories")
    # Member 2's interface: the query is inst['prefix'] only, candidates are the history.
    demos = retrieve_demonstrations(inst, history, idf, method=retriever, top_k=top_k)
    blocks, shown = [], []
    for i, d in enumerate(demos, 1):
        ctx = d["context"][-C.DEMO_MAX_CONTEXT:]
        shown += ctx + [d["next"]]
        blocks.append(f"Example {i}:\n{_lines(ctx, name_fn)}\n"
                      f"  -> next: {format_checkin(d['next'], name_fn)}")
    prefix_block, prefix = _prefix_block(inst, name_fn)
    user = (
        f"Each check-in is written as: {_format_line(name_fn)}.\n\n"
        "<examples>\nEarlier trajectories of this user that are most similar to the current one, "
        "most similar first. Each ends with the POI the user visited next.\n"
        + "\n".join(blocks) +
        "\n</examples>\n\n"
        f"{prefix_block}\n\n"
        "Use the examples to see where this user tends to go after similar sequences.\n"
        f"{_question(inst)}"
    )
    meta = _meta("retrieval", inst, shown + prefix, name_fn, retriever=retriever,
                 demo_traj_ids=[d["traj_id"] for d in demos],
                 demo_scores=[d["score"] for d in demos])
    return Prompt(SYSTEM_PROMPT, user, meta)


def build_prompt(method: str, inst: dict, history: list[dict], idf=None,
                 retriever: str = C.DEFAULT_RETRIEVER, name_fn: NameFn | None = None) -> Prompt:
    if method == "llmmob":
        return build_llmmob(inst, history, name_fn)
    if method == "retrieval":
        return build_retrieval(inst, history, idf, retriever, name_fn=name_fn)
    raise ValueError(f"unknown method {method!r}; expected one of {C.METHODS}")
