"""E3 name test: consistent fake venue aliases and the perturbed test sets (Member 3).

    from tracex_llm.aliasing import AliasMap
    am = AliasMap.for_dataset("msteps_tokyo")
    build_prompt(..., name_fn=am.name_fn)       # "aliased" condition

    python -m tracex_llm.aliasing export        # writes data/perturbed/msteps_tokyo/ (release deliverable)

Design (stated in the paper):
- One alias per poi_id, fixed by the seed, used everywhere the POI appears (history, demonstrations,
  prefix). History, targets, categories, times and POI ids are identical across the three conditions,
  so any accuracy change comes from the names alone.
- Aliases are per poi_id, not per name: two branches of one chain get different aliases, so the
  chain information carried by real names is removed too.
- Real Massive-STEPS Tokyo names are 95% Japanese script (median 9 characters). Aliases are random
  katakana pseudo-words of similar length, so "aliased" differs from "present" in meaning, not in
  script or length. Each alias is unique and never equals or occurs inside a real venue name.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from tracex_data.api import load_dataset

from . import config as C

ALIAS_VERSION = "katakana-v1"   # bump if the alias generator changes (enters the run fingerprint)

# Plain katakana morae (no small kana), plus the long-vowel mark used inside words.
_MORAE = list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
              "ガギグゲゴザジズゼゾダデドバビブベボパピプペポ")
_LONG = "ー"
MIN_MORAE, MAX_MORAE = 5, 9


def _alias(rng: random.Random) -> str:
    n = rng.randint(MIN_MORAE, MAX_MORAE)
    out = []
    for i in range(n):
        m = rng.choice(_MORAE)
        if m == "ン" and (i == 0 or out[-1] == "ン"):   # no word-initial or doubled ン
            m = rng.choice(_MORAE[:-1])
        out.append(m)
        if 0 < i < n - 1 and rng.random() < 0.12:
            out.append(_LONG)
    return "".join(out)


@dataclass
class AliasMap:
    dataset: str
    aliases: dict[int, str]
    seed: int = C.SEED

    @classmethod
    def build(cls, dataset: str, pois, seed: int = C.SEED) -> "AliasMap":
        real = {str(x) for x in pois["name"].dropna()}
        rng = random.Random(f"{seed}-{dataset}-{ALIAS_VERSION}")
        used, aliases = set(), {}
        for pid in sorted(int(p) for p in pois["poi_id"]):
            while True:
                a = _alias(rng)
                if a not in used and not any(a in r for r in real):
                    break
            used.add(a)
            aliases[pid] = a
        return cls(dataset, aliases, seed)

    @classmethod
    def for_dataset(cls, dataset: str, seed: int = C.SEED) -> "AliasMap":
        if dataset not in C.NAME_DATASETS:
            raise ValueError(f"{dataset} has no venue names; the name test runs on {C.NAME_DATASETS}")
        return cls.build(dataset, load_dataset(dataset).pois, seed)

    def name_fn(self, checkin: dict) -> str:
        return self.aliases[checkin["poi_id"]]


# ---------------------------------------------------------------------------
# Condition transforms and export of the perturbed test sets
# ---------------------------------------------------------------------------
def name_fn_for(dataset: str, condition: str):
    """Name function for an E3 condition (None = no names shown)."""
    from .prompts import present_names
    if condition in (C.NO_NAMES, "removed"):
        return None
    if condition == "present":
        return present_names
    if condition == "aliased":
        return AliasMap.for_dataset(dataset).name_fn
    raise ValueError(f"unknown condition {condition!r}")


def transform_checkin(c: dict, name_fn) -> dict:
    out = {k: v for k, v in c.items() if k != "name"}
    if name_fn is not None:
        out["name"] = name_fn(c)
    return out


def _traj(t: dict, name_fn) -> dict:
    return {**t, "checkins": [transform_checkin(c, name_fn) for c in t["checkins"]]}


def export(dataset: str = "msteps_tokyo", sample: str = "sample_test_500"):
    """Self-contained perturbed test sets: one JSONL per condition, history embedded."""
    ds = load_dataset(dataset)
    am = AliasMap.for_dataset(dataset)
    out = C.PERTURBED_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    insts = ds.instances(sample)
    fns = {"present": lambda c: c.get("name"), "removed": None, "aliased": am.name_fn}
    for cond, fn in fns.items():
        with open(out / f"{sample}_{cond}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for inst in insts:
                rec = {**inst,
                       "prefix": [transform_checkin(c, fn) for c in inst["prefix"]],
                       "target": transform_checkin(inst["target"], fn),
                       "history": [_traj(t, fn) for t in ds.history(inst)],
                       "name_condition": cond}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(out / "alias_map.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("poi_id,alias\n")
        for pid, a in sorted(am.aliases.items()):
            f.write(f"{pid},{a}\n")
    (out / "README.md").write_text(README.format(dataset=dataset, sample=sample, n=len(insts),
                                                 seed=am.seed, version=ALIAS_VERSION),
                                   encoding="utf-8", newline="\n")
    return out


README = """# TraceX perturbed test sets: {dataset} ({sample})

The E3 name test of TraceX ("Copy, Recall, or Reason?"). {n} test instances in three versions that differ
**only in venue names**. History, targets, POI ids, categories and times are identical.

| File | Venue names |
|---|---|
| `{sample}_present.jsonl` | real Massive-STEPS names |
| `{sample}_removed.jsonl` | no `name` field |
| `{sample}_aliased.jsonl` | consistent fake aliases from `alias_map.csv` |

Each line is one instance: `prefix` (input check-ins), `target` (the check-in to predict), `history`
(the user's earlier training trajectories, oldest first) and the scoring flags `target_in_history`,
`target_in_prefix` and `is_revisit`. Never put `target` or the flags in a prompt.

Aliases: one per `poi_id` (seed {seed}, generator `{version}`), random katakana pseudo-words similar in
length and script to the real names. Each is unique and never equals or occurs inside a real venue name.
Chain branches get different aliases, so chain identity is removed as well.

Regenerate with `python -m tracex_llm.aliasing export`.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tracex_llm.aliasing")
    ap.add_argument("command", choices=["export", "show"])
    ap.add_argument("--dataset", default="msteps_tokyo", choices=list(C.NAME_DATASETS))
    ap.add_argument("--sample", default="sample_test_500")
    a = ap.parse_args(argv)
    if a.command == "export":
        print(f"[aliasing] wrote {export(a.dataset, a.sample)}")
    else:
        ds = load_dataset(a.dataset)
        am = AliasMap.build(a.dataset, ds.pois)
        for _, r in ds.pois.head(15).iterrows():
            print(f"{r.poi_id:5d}  {r['name']}  ->  {am.aliases[int(r.poi_id)]}")


if __name__ == "__main__":
    main()
