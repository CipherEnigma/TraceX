"""Assistant LLM layer (tracex_assist/llm.py, contract.py): no Ollama, no dataset needed."""
import urllib.error

import pytest

from tracex_assist import llm
from tracex_assist.contract import (Context, Evidence, Explore, Fact, Place, Suggestion, category_matches, routine_key,
                                    time_bucket, to_dict)


class FakePlaces:
    P = {1: Place(1, "Gym / Fitness Center", None, 40.0, -74.0, 50),
         2: Place(2, "Fast Food Restaurant", None, 40.001, -74.0, 90),
         3: Place(3, "Coffee Shop", None, 40.002, -74.0, 30),
         7: Place(7, "Burger Joint", None, 40.003, -74.0, 140),
         8: Place(8, "Ramen Restaurant", "長浜らーめん", 40.004, -74.0, 12)}

    def get(self, pid):
        return self.P[pid]


def evidence(want_new=False, thin=False, habits=True):
    f1 = Fact("transition", 2, "Fast Food Restaurant", 11, 20, "you, after this place, on Fridays")
    f2 = Fact("frequency", 3, "Coffee Shop", 4, 20, "you, all visits")
    return Evidence(
        Context(weekday=4, hour=18, current_poi=1, want_new=want_new),
        [Suggestion(2, 0.9, [f1]), Suggestion(3, 0.3, [f2])] if habits else [],
        [("Fast Food Restaurant", 0.55)],
        [Explore(7, 0.3, 140, "Burger Joint"), Explore(8, 0.6, 12, "Ramen Restaurant")],
        [f1, f2], {"acc1": 0.25, "acc5": 0.6, "n": 500}, thin)


P = FakePlaces()


# ---------------------------------------------------------------- contract helpers
def test_time_buckets_wrap_midnight():
    assert [time_bucket(h) for h in (23, 0, 4, 5, 12, 15, 18)] == \
        ["night", "night", "night", "morning", "midday", "afternoon", "evening"]
    assert routine_key(4, 18) == "Fri|evening"


def test_category_kinds():
    assert category_matches("food", "Ramen Restaurant") and category_matches("coffee", "Café")
    assert not category_matches("shopping", "Coffee Shop")  # coffee, not shopping
    assert not category_matches("outdoors", "Beer Garden") and category_matches("drinks", "Beer Garden")
    assert not category_matches("transport", "Police Station") and category_matches("transport", "Subway")
    assert category_matches(None, "anything")


def test_to_dict_adds_share():
    d = to_dict(evidence())
    assert d["facts"][0]["share"] == 0.55 and d["context"]["weekday"] == 4


# ---------------------------------------------------------------- intent
@pytest.mark.parametrize("q,new,kind", [
    ("suggest somewhere new to eat", True, "food"),
    ("where do I usually get coffee?", False, "coffee"),
    ("a coffee shop I have never been to", True, "coffee"),  # coffee wins over shopping
    ("where next?", False, None),
    ("any bar I haven't tried?", True, "drinks"),
])
def test_parse_intent(q, new, kind):
    assert llm.parse_intent(q) == {"want_new": new, "want_kind": kind}


@pytest.mark.parametrize("q,kind", [  # the place the user is at / coming from is not the request
    ("I just left the bar. where to eat?", "food"),
    ("just got off the train", None),
    ("where can I eat after the gym?", "food"),
    ("I am at the station, any coffee nearby?", "coffee"),
    ("coming from the park, want drinks", "drinks"),
    ("back from the coffee shop, what now?", None),
    ("I am in the mall, where can I buy groceries", "shopping"),
    ("take me to a train station", "transport"),
    ("I want to go to a park", "outdoors"),
    ("lunch in shibuya?", "food"),
])
def test_kind_ignores_where_the_user_is(q, kind):
    assert llm.wanted_kind(q) == kind


# ---------------------------------------------------------------- view
def test_view_has_exact_numbers_and_labels():
    v = llm.evidence_view(evidence(), P)
    assert v["likely_next"][0]["place"] == "Fast Food Restaurant #2"
    assert "11 of 20 times (55%)" in v["likely_next"][0]["why"][0]
    assert v["situation"].startswith("Fri 18:00 (evening), currently at Gym / Fitness Center #1")
    assert "accuracy" in v and "25%" in v["accuracy"]


def test_view_shows_new_places_only_when_asked():
    assert llm.evidence_view(evidence(want_new=False), P)["new_places"] == []
    v = llm.evidence_view(evidence(want_new=True), P)
    assert [p["place"] for p in v["new_places"]] == ["Burger Joint #7", "長浜らーめん (Ramen Restaurant #8)"]
    assert llm.evidence_view(evidence(habits=False), P)["new_places"]  # nothing else to offer


# ---------------------------------------------------------------- validator
def test_validator_accepts_grounded_answer():
    v = llm.evidence_view(evidence(want_new=True), P)
    text = ("After the gym you went to Fast Food Restaurant #2 11 of 20 times (55%). "
            "For something new, try Burger Joint #7, 0.3 km away, or 長浜らーめん at 0.6 km.")
    assert llm.validate(text, v) == []


@pytest.mark.parametrize("text,problem", [
    ("Go to Pizza Place #99.", "places not in the evidence"),
    ("You go there 70% of the time (Fast Food Restaurant #2).", "percentages not in the evidence"),
    ("You went there 12 out of 20 times.", "counts not in the evidence"),
    ("Burger Joint #7 is 2.5 km away.", "distances not in the evidence"),
    ("Fast Food Restaurant #2 is 0.3 km away.", "not next to the place it belongs to"),
    ("", "empty answer"),
])
def test_validator_rejects_invented_content(text, problem):
    v = llm.evidence_view(evidence(want_new=True), P)
    assert any(problem in p for p in llm.validate(text, v))


# ---------------------------------------------------------------- answer() control flow
def test_answer_accepts_first_valid_draft(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda m, model: "You usually go to Fast Food Restaurant #2 (11 of 20 times (55%)).")
    r = llm.answer("where next?", evidence(), P)
    assert r.source == "llm" and r.problems == []


def test_answer_retries_once_then_accepts(monkeypatch):
    drafts = iter(["Go to #99, 80% of the time.", "Go to Fast Food Restaurant #2."])
    seen = []
    monkeypatch.setattr(llm, "chat", lambda m, model: (seen.append(m[-1]["content"]), next(drafts))[1])
    r = llm.answer("where next?", evidence(), P)
    assert r.source == "llm-retry" and "broke the rules" in seen[1] and r.problems


def test_answer_falls_back_to_template_after_two_bad_drafts(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda m, model: "Go to #99.")
    r = llm.answer("where next?", evidence(), P)
    assert r.source == "template" and "Fast Food Restaurant #2" in r.text


def test_answer_without_ollama_uses_template(monkeypatch):
    def down(m, model):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(llm, "chat", down)
    r = llm.answer("somewhere new?", evidence(), P)
    assert r.source == "template" and "Burger Joint #7" in r.text and any("Ollama" in p for p in r.problems)


def test_template_is_itself_grounded():
    for ev, q in ((evidence(), "where next?"), (evidence(want_new=True), "new place?"), (evidence(thin=True), ""),
                  (evidence(habits=False), "")):
        text = llm.fallback_answer(ev, P, q)
        assert llm.validate(text, llm.evidence_view(ev, P)) == [], text


def test_prompt_contains_only_the_evidence():
    v = llm.evidence_view(evidence(), P)
    msgs = llm.build_messages("where next?", v)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert "Burger Joint" not in msgs[1]["content"]  # new places hidden unless asked for


def test_ollama_status_never_raises(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens there
    s = llm.ollama_status()
    assert s["running"] is False and s["error"]


def test_requested_kind_narrows_known_places():
    ev = evidence()
    ev.context.want_kind = "coffee"  # the user knows Coffee Shop #3 -> only that one is offered
    v = llm.evidence_view(ev, P)
    assert [x["place"] for x in v["likely_next"]] == ["Coffee Shop #3"] and v["new_places"] == []
    ev.context.want_kind = "food"  # Fast Food Restaurant #2 is food; Coffee Shop is not
    assert [x["place"] for x in llm.evidence_view(ev, P)["likely_next"]] == ["Fast Food Restaurant #2"]
    ev.context.want_kind = "drinks"  # no known bar -> new places of that kind (none here) and no habits
    v = llm.evidence_view(ev, P)
    assert v["likely_next"] == [] and v["new_places"] == []
    assert llm.validate(llm.fallback_answer(ev, P, "a bar?"), v) == []


def test_filler_sentences_are_removed():
    t = "Go to Chinese Restaurant #332 (3 of 3 times (100%)). You will enjoy it."
    assert llm.strip_filler(t) == "Go to Chinese Restaurant #332 (3 of 3 times (100%))."
    assert llm.strip_filler("Great choice!") == "Great choice!"  # never return an empty answer
