"""Unit tests for the TraceX LLM pipeline (Member 3) on small synthetic data.

No GPU, model download or processed dataset is needed: the runner is tested with the mock
backend and an in-memory dataset.

Run:  python -m pytest tests/test_llm.py -q
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from tracex_ir.retrieval import IDF
from tracex_llm import config as C
from tracex_llm import run as R
from tracex_llm import stats as S
from tracex_llm.aliasing import AliasMap, transform_checkin
from tracex_llm.evaluate import compare, score, summarize
from tracex_llm.parse import parse_response, user_poi_ids
from tracex_llm.prompts import build_prompt, present_names, recent_history


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def ck(poi, t="2018-09-08 11:56:00", wd=5, cat="Cafe", name=None):
    c = {"poi_id": poi, "category": cat, "time": t, "hour": int(t[11:13]), "weekday": wd}
    if name is not None:
        c["name"] = name
    return c


NAMES = {1: "カフェ一号", 2: "駅前ラーメン", 3: "中央公園", 4: "書店本店", 5: "ジム", 99: "秘密の店"}
HIST = [
    {"traj_id": 1, "user_id": 7, "split": "train", "checkins": [ck(1, name=NAMES[1]), ck(2, name=NAMES[2])]},
    {"traj_id": 2, "user_id": 7, "split": "train", "checkins": [ck(3, name=NAMES[3]), ck(1, name=NAMES[1]), ck(4, name=NAMES[4])]},
    {"traj_id": 3, "user_id": 7, "split": "train", "checkins": [ck(2, name=NAMES[2]), ck(5, name=NAMES[5])]},
]


def instance(target_poi=99, iid="synth-test-1"):
    return {"instance_id": iid, "dataset": "synth", "split": "test", "user_id": 7, "traj_id": 10,
            "history_traj_ids": [1, 2, 3], "prefix": [ck(1, name=NAMES[1]), ck(3, name=NAMES[3])],
            "target": ck(target_poi, t="2018-09-08 17:41:00", cat="Secret Bar", name=NAMES.get(target_poi, "X")),
            "target_in_history": False, "target_in_prefix": False, "is_revisit": False,
            "n_history_trajs": 3, "n_history_checkins": 7}


IDF_ = IDF(HIST)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", C.METHODS)
@pytest.mark.parametrize("names", [None, present_names])
def test_prompt_never_contains_target(method, names):
    inst = instance()
    p = build_prompt(method, inst, HIST, IDF_, name_fn=names).user
    leaked = copy.deepcopy(inst)
    leaked["target"].update(poi_id=424242, category="LEAKCAT", name="LEAKNAME")
    leaked.update(target_in_history=True, target_in_prefix=True, is_revisit=True)
    q = build_prompt(method, leaked, HIST, IDF_, name_fn=names).user
    assert p == q, "changing the target's POI, category, name or flags must not change the prompt"
    assert "LEAK" not in q and "424242" not in q and "Secret Bar" not in q


def test_prompt_states_target_time_only_when_enabled(monkeypatch):
    inst = instance()
    assert "2018-09-08 17:41" in build_prompt("llmmob", inst, HIST).user
    monkeypatch.setattr(C, "INCLUDE_TARGET_TIME", False)
    assert "17:41" not in build_prompt("llmmob", inst, HIST).user


def test_prompt_names_and_poi_meta():
    inst = instance()
    with_names = build_prompt("llmmob", inst, HIST, name_fn=present_names)
    without = build_prompt("llmmob", inst, HIST)
    assert NAMES[1] in with_names.user and NAMES[1] not in without.user
    assert set(with_names.meta["prompt_poi_ids"]) == {1, 2, 3, 4, 5}
    assert with_names.messages()[0]["role"] == "system"


def test_recent_history_budget_keeps_newest():
    groups = recent_history(HIST, max_checkins=3)
    shown = [c["poi_id"] for _, ch in groups for c in ch]
    assert shown == [4, 2, 5]  # newest 3 check-ins, still oldest-first


def test_retrieval_prompt_uses_demonstrations():
    p = build_prompt("retrieval", instance(), HIST, IDF_, retriever="jaccard")
    assert "-> next:" in p.user and p.meta["retriever"] == "jaccard" and p.meta["demo_traj_ids"]
    with pytest.raises(ValueError):
        build_prompt("retrieval", instance(), HIST, None, retriever="bm25")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
VALID = set(range(1, 20))


@pytest.mark.parametrize("text,status,ranked", [
    ('{"prediction": [3, 7, 3, 8], "reason": "habit"}', "json", [3, 7, 8]),
    ('Sure!\n```json\n{"prediction": ["POI 5", "6"], "reason": "x"}\n```', "json", [5, 6]),
    ("{'prediction': [4, 2]}", "literal", [4, 2]),
    ('{"prediction": [{"poi_id": 2}, {"poi_id": 9}]}', "json", [2, 9]),
    ('{"prediction": [5, 6, 7,, ], "reason": "cut', "regex_list", [5, 6, 7]),
    ("I think POI 12, then POI 1.", "poi_mentions", [12, 1]),
    ('{"prediction": []}', "empty", []),
    ("I cannot answer.", "failed", []),
])
def test_parse_formats(text, status, ranked):
    r = parse_response(text, VALID)
    assert r.status == status and r.ranked == ranked
    assert r.invalid_answer == (not ranked)


def test_parse_keeps_invalid_ids_in_place_and_caps():
    r = parse_response('{"prediction": [3, 99999, 4]}', VALID, input_ids={4}, prompt_ids={3})
    assert r.ranked == [3, 99999, 4] and r.invalid_ids == [99999]
    assert r.top1 == 3 and not r.invalid_answer and r.copied_top1 is False and r.copied_top1_prompt is True
    long = parse_response(json.dumps({"prediction": list(range(1, 30))}), VALID)
    assert len(long.ranked) == C.N_PREDICTIONS
    bad_top = parse_response('{"prediction": [99999, 3]}', VALID)
    assert bad_top.invalid_answer  # no usable top-1


def test_user_poi_ids_excludes_target():
    assert user_poi_ids(instance(target_poi=99), HIST) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# E3 aliasing
# ---------------------------------------------------------------------------
POIS = pd.DataFrame({"poi_id": list(NAMES), "name": list(NAMES.values())})


def test_aliases_deterministic_unique_and_never_real():
    a, b = AliasMap.build("synth", POIS), AliasMap.build("synth", POIS)
    assert a.aliases == b.aliases
    vals = list(a.aliases.values())
    assert len(set(vals)) == len(vals) == len(POIS)
    assert not any(v in r for v in vals for r in NAMES.values())
    assert AliasMap.build("synth", POIS, seed=7).aliases != a.aliases


def test_conditions_differ_only_in_names():
    am = AliasMap.build("synth", POIS)
    inst = instance()
    prompts = {c: build_prompt("llmmob", inst, HIST, name_fn=f).user
               for c, f in (("present", present_names), ("removed", None), ("aliased", am.name_fn))}
    import re
    strip = lambda s: re.sub(r' \| "[^"]*"', "", s)  # noqa: E731
    assert len({strip(p) for p in prompts.values()}) == 1
    assert not any(n in prompts["aliased"] or n in prompts["removed"] for n in NAMES.values())


def test_transform_checkin():
    c = ck(1, name=NAMES[1])
    assert "name" not in transform_checkin(c, None)
    assert transform_checkin(c, lambda x: "ALIAS")["name"] == "ALIAS"
    assert c["name"] == NAMES[1]  # input untouched


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_mcnemar_exact_and_chi2():
    a = [1] * 10 + [0] * 2 + [1] * 5
    b = [0] * 10 + [1] * 2 + [1] * 5
    r = S.mcnemar(a, b)
    assert r["test"] == "exact" and abs(r["p"] - 158 / 4096) < 1e-12
    r = S.mcnemar([1] * 30 + [0] * 15, [0] * 30 + [1] * 15)
    assert r["test"] == "chi2-cc" and abs(r["p"] - 0.036888) < 1e-5
    assert S.mcnemar([1, 0], [1, 0])["p"] == 1.0


def test_holm_and_bootstrap():
    adj = S.holm([0.01, 0.04, 0.03, float("nan")])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06]) and math.isnan(adj[3])
    x = np.r_[np.ones(30), np.zeros(70)]
    ci = S.bootstrap_ci(x)
    assert ci["mean"] == 0.3 and ci["lo"] < 0.3 < ci["hi"]
    same = S.paired_bootstrap(x, x)
    assert same["diff"] == 0 and same["p"] == 1.0
    with pytest.raises(ValueError):
        S.paired_bootstrap([1, 0], [1])


def test_score_and_compare():
    s = score([5, 9, 3], 3)
    assert s["acc1"] == 0 and s["acc5"] == 1 and s["mrr"] == pytest.approx(1 / 3)
    s = score([5, 9, 3], 3, {5: "Bar", 9: "Gym", 3: "Cafe"}, "Gym")
    assert s["cat_acc1"] == 0 and s["cat_acc5"] == 1
    row = lambda v: {"acc1": v, "acc5": v, "mrr": v, "ndcg5": v, "cat_acc1": v, "cat_acc5": v,  # noqa: E731
                     "invalid": 0.0, "copy": 1.0,
                     "copy_prompt": 1.0, "is_revisit": True, "target_in_prompt": True, "status": "json",
                     "seconds": 1.0, "new_tokens": 10}
    a = {f"i{k}": row(1.0) for k in range(10)}
    b = {f"i{k}": row(0.0) for k in range(10)}
    c = compare(a, b, n_boot=200)
    assert c["n"] == 10 and c["acc1"]["diff"] == 1.0 and c["mcnemar_acc1"]["a_only"] == 10
    s = summarize(a, expected_n=12, n_boot=200)
    assert s["n"] == 10 and not s["complete"] and s["revisit"]["n"] == 10 and s["new"]["n"] == 0


# ---------------------------------------------------------------------------
# Runner: resume, corruption recovery, settings guard (mock backend, in-memory data)
# ---------------------------------------------------------------------------
class FakeDataset:
    def __init__(self):
        self.insts = [instance(target_poi=t, iid=f"synth-test-{k}") for k, t in enumerate([1, 2, 3, 4, 5, 99])]
        self.pois = POIS

    def instances(self, which):
        return self.insts

    def history(self, inst):
        return HIST

    def train_trajectories(self):
        return HIST


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(R, "load_dataset", lambda name: FakeDataset())
    return R


def test_runner_resume_and_recovery(runner):
    args = ["--dataset", "nyc", "--model", "mock", "--method", "llmmob", "--sample", "sample_test_debug20"]
    first = runner.main(args + ["--limit", "4"])
    path = C.result_path("nyc", "mock", "llmmob", sample="sample_test_debug20")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    runner.main(args)  # resumes: only the 2 missing instances are run
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and len({json.loads(x)["instance_id"] for x in lines}) == 6
    rec = json.loads(lines[0])
    assert rec["target"] == 1 and rec["parse"]["status"] == "json" and "prompt_sha1" in rec
    assert first[list(first)[0]]["n"] == 4

    path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20], encoding="utf-8")  # truncated last line
    runner.main(args)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and all(json.loads(x) for x in lines)


def test_runner_refuses_changed_settings(runner, monkeypatch):
    args = ["--dataset", "nyc", "--model", "mock", "--method", "llmmob", "--sample", "sample_test_debug20"]
    runner.main(args + ["--limit", "2"])
    monkeypatch.setattr(C, "N_PREDICTIONS", 5)
    with pytest.raises(SystemExit, match="settings changed"):
        runner.main(args)
    runner.main(args + ["--overwrite"])  # explicit restart is allowed


def test_run_name_validation():
    assert C.run_name("nyc", "qwen7b", "retrieval") == "nyc__qwen7b__retrieval-jaccard__none__sample_test_500"
    with pytest.raises(ValueError):
        C.run_name("nyc", "qwen7b", "llmmob", "aliased")
    with pytest.raises(ValueError):
        C.run_name("nyc", "gpt", "llmmob")
