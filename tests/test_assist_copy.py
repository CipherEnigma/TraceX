"""Tests for the copy analysis (Member 1, task 1.4) on synthetic data.

Run:  python -m pytest tests/test_assist_copy.py -q
"""
from __future__ import annotations

import copy
import json

import pytest

from tracex_assist import analysis_copy as A
from tracex_ir.baselines import GlobalPopularity


def ck(poi, day, hour):
    return {"poi_id": poi, "category": "x", "time": f"2013-01-{day:02d} {hour:02d}:00:00", "hour": hour, "weekday": 0}


HISTORY = [{"traj_id": 1, "checkins": [ck(1, 1, 9), ck(2, 1, 10), ck(1, 1, 11), ck(2, 1, 12)]},
           {"traj_id": 2, "checkins": [ck(3, 3, 9), ck(4, 3, 10), ck(3, 3, 12)]}]


def instance(iid="a", prefix=(1,), target=2, revisit=True):
    return {"instance_id": iid, "prefix": [ck(p, 9, 8 + i) for i, p in enumerate(prefix)], "target": ck(target, 9, 22),
            "is_revisit": revisit, "n_history_checkins": 7}


class FakeData:
    def __init__(self, insts):
        self.insts = insts

    def train_trajectories(self):
        return HISTORY

    def instances(self, which):
        return self.insts

    def history(self, inst):
        return HISTORY


def record(iid, target, ranked, invalid=False, model="qwen7b", condition="none"):
    return {"instance_id": iid, "dataset": "synth", "model": model, "method": "llmmob", "retriever": None,
            "condition": condition, "target": target, "parse": {"ranked": ranked, "invalid_answer": invalid}}


INSTANCES = [instance("a", (1,), 2), instance("b", (3, 4), 9, revisit=False), instance("c", (2,), 1)]


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "load_dataset", lambda name: FakeData(INSTANCES))

    def save(rows, name="run", tail=""):
        (tmp_path / "synth").mkdir(exist_ok=True)
        (tmp_path / "synth" / f"{name}__s.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows) + tail, encoding="utf-8")
        return tmp_path
    return save


def test_rules():
    g = A.rule_guesses(instance(prefix=(1,)), HISTORY, GlobalPopularity(HISTORY))
    assert (g["last"], g["markov"], g["counter"], g["trip"]) == (1, 2, 1, {1})
    assert g["markov_tied"] == {2} and g["own"] == {1, 2, 3, 4}


def test_markov_has_no_fallback_and_ties_are_reported():
    pop = GlobalPopularity(HISTORY)
    assert A.rule_guesses(instance(prefix=(99,)), HISTORY, pop)["markov"] is None
    g = A.rule_guesses(instance(prefix=(4,)), [{"checkins": [ck(4, 1, 1), ck(5, 1, 2), ck(4, 1, 3), ck(6, 1, 4)]}],
                       pop)
    assert g["markov"] == 5 and g["markov_tied"] == {5, 6}


def test_markov_does_not_join_separate_trips():
    trips = [{"checkins": [ck(8, 1, 1), ck(7, 1, 2)]}, {"checkins": [ck(6, 2, 1), ck(5, 2, 2)]}]
    assert A.rule_guesses(instance(prefix=(7,)), trips, GlobalPopularity(trips))["markov"] is None


def test_rules_never_read_the_target_place():
    inst = instance(target=2)
    other = copy.deepcopy(inst)
    other["target"].update(poi_id=777, category="secret")
    other["is_revisit"] = False
    pop = GlobalPopularity(HISTORY)
    assert A.rule_guesses(inst, HISTORY, pop) == A.rule_guesses(other, HISTORY, pop)


def test_history_only_comparator_ignores_the_current_trip():
    pop = GlobalPopularity(HISTORY)
    a, b = (A.rule_guesses(instance(prefix=p), HISTORY, pop) for p in ((4, 4, 4), (1,)))
    assert a["heuristic_history"] == b["heuristic_history"] and a["heuristic"] != b["heuristic"]


@pytest.mark.parametrize("n,label", [(1, "1-9"), (9, "1-9"), (10, "10-40"), (40, "10-40"), (41, "41-150"),
                                     (150, "41-150"), (151, "151+")])
def test_history_bins(n, label):
    assert A.history_bin(n) == label


def test_shares_and_accuracy(results):
    d = results([record("a", 2, [2]), record("b", 9, [3]), record("c", 1, [], invalid=True)])
    res = A.analyse(d, "s", n_boot=200)
    r = res["runs"]["run__s"]
    assert r["n"] == 3 and r["complete"] and r["main"] and r["acc1"] == pytest.approx(1 / 3)
    assert r["match"]["markov"]["mean"] == pytest.approx(2 / 3)  # 1 -> 2 and 4 -> 3
    assert r["match"]["last"]["mean"] == 0
    assert r["match"]["trip"]["mean"] == pytest.approx(1 / 3)    # 3 is in trip b
    assert r["new_hits"] == {"hits": 0, "n": 1}
    # every guess is explained by a rule, another of the user's places, a never-seen place, or invalid
    assert r["match"]["any_rule"]["mean"] + sum(r["outside"].values()) == pytest.approx(1)


def test_an_invalid_poi_matches_nothing(results):
    d = results([record("a", 2, [123456], invalid=True)])
    r = A.analyse(d, "s", n_boot=100)["runs"]["run__s"]
    assert r["outside"] == {"other_own": 0, "never_seen": 0, "invalid": 1} and not r["complete"]


def test_truncated_line_and_mock_runs_are_skipped(results):
    results([record("a", 2, [2])], name="real", tail='{"instance_id": "b", "tar')
    d = results([record("a", 2, [2], model="mock")], name="fake")
    runs = A.analyse(d, "s", n_boot=100)["runs"]
    assert list(runs) == ["real__s"] and runs["real__s"]["n"] == 1


@pytest.mark.parametrize("row", [record("a", 5, [2]), record("zzz", 2, [2])])
def test_results_that_do_not_fit_the_data_are_refused(results, row):
    with pytest.raises(ValueError):
        A.analyse(results([row]), "s", n_boot=100)


def test_no_results_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.analyse(tmp_path, "s")


def test_holm_families():
    def run(main, *ps):
        cells = {name: {"p": p, "diff": 0.1} for name, p in zip(("all", "morning", "night", "new"), ps)}
        return {"main": main, "vs": {"counter": cells}}

    runs = {"a": run(True, 0.03, 0.01, 0.04, 0.02), "b": run(False, 0.03, 0.01, 0.04, 0.02)}
    A.adjust(runs)
    c = runs["a"]["vs"]["counter"]
    assert [c[k]["p_holm"] for k in ("all", "morning", "night", "new")] == pytest.approx([0.03, 0.02, 0.04, 0.02])
    assert c["all"]["beats"]


def test_outputs_are_written_as_strict_json_and_markdown(results, tmp_path):
    d = results([record("a", 2, [2]), record("b", 9, [3]), record("c", 1, [2])])
    out = tmp_path / "out"
    A.main(["--results-dir", str(d), "--sample", "s", "--out-dir", str(out), "--n-boot", "100"])
    json.loads((out / "copy_analysis_s.json").read_text(encoding="utf-8"),
               parse_constant=lambda x: pytest.fail(f"{x} in JSON"))
    md = (out / "copy_analysis_s.md").read_text(encoding="utf-8")
    assert "## 1." in md and "## 5." in md
