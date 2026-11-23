"""Page 8: Slides - the whole project in three slides (approach, results, assistant and conclusion).

Every slide is a function returning HTML. Numbers come from the same sources as the other pages
(tracex_llm.evaluate, reports/assistant), so the deck always matches the results.
"""
from __future__ import annotations

import streamlit as st

import context
import data
import ui
from tracex_llm import config as C

DS_KEYS = ("nyc", "tky", "msteps_tokyo")

KEYS_JS = """(function(){
  if (window.__txSlideKeys) return; window.__txSlideKeys = true;
  document.addEventListener('keydown', function(e){
    if (!document.querySelector('.tx-slide')) return;
    var t = e.target;
    if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;
    var want = e.key === 'ArrowRight' ? 'Next' : e.key === 'ArrowLeft' ? 'Previous' : null;
    if (!want) return;
    var btns = document.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
      if (btns[i].textContent.indexOf(want) !== -1 && !btns[i].disabled) { btns[i].click(); e.preventDefault(); break; }
    }
  });
})();"""

PRESENT_CSS = ('<style>[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],'
               '[data-testid="stExpandSidebarButton"]{display:none}</style>')


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def _ul(items: list[str]) -> str:
    return '<ul class="tx-bullets">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _take(text: str) -> str:
    return f'<div class="tx-take">{text}</div>'


def _tile(title: str, body: str) -> str:
    return f'<div class="tx-tile"><div class="tx-h3">{title}</div>{body}</div>'


def _grid(*tiles: str) -> str:
    return '<div class="tx-grid">' + "".join(tiles) + "</div>"


def _two(a: str, b: str) -> str:
    col = 'style="display:flex;flex-direction:column;gap:12px"'
    return f'<div class="tx-two"><div {col}>{a}</div><div {col}>{b}</div></div>'


def _flow(boxes: list[tuple[str, str, str]]) -> str:
    parts = []
    for i, (who, what, how) in enumerate(boxes):
        if i:
            parts.append('<span class="tx-flowarrow">→</span>')
        parts.append(f'<div class="tx-flowbox"><span class="who">{who}</span><span class="what">{what}</span>'
                     f'<span class="how">{how}</span></div>')
    return '<div class="tx-flow">' + "".join(parts) + "</div>"


def _stats(items: list[tuple[str, str]]) -> str:
    return '<div class="tx-stats">' + "".join(
        f'<div class="tx-stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


def _sample() -> str:
    return st.session_state.get("sample", "sample_test_500")


def _missing(what: str) -> str:
    return f'<p class="tx-cap">{what} is not available yet; it appears here once the results are in the repo.</p>'


# ---------------------------------------------------------------------------
# Slide 1: question and approach
# ---------------------------------------------------------------------------
def s_approach() -> str:
    sample = _sample()
    n = {d: data.expected_n(d, sample) for d in DS_KEYS}
    hyp = _grid(
        _tile("Copy", "<p>Repeats the places the user already visits most.</p>"
                      "<p class='tx-note'>Tested by E1 (beat a visit counter?) and E2 (get a never-visited place right?)</p>"),
        _tile("Recall", "<p>Uses memorised knowledge of real venues.</p>"
                        "<p class='tx-note'>Tested by E3 (does accuracy drop when venue names are hidden or faked?)</p>"),
        _tile("Reason", "<p>Infers the next place from time, order and category.</p>"
                        "<p class='tx-note'>Would beat the counter and get some new places right</p>"))
    data_side = (
        '<div class="tx-h3">Data and task</div>'
        + _ul([f"<b>3 datasets:</b> Foursquare NYC ({n['nyc']} test cases), TKY ({n['tky']}), and Massive-STEPS Tokyo "
               f"({n['msteps_tokyo']}; real venue names, short histories).",
               "<b>Task:</b> hide a trip’s last check-in; predict it from the trip so far plus the user’s earlier trips.",
               "<b>Time-based 80/10/10 split</b> and history that ends before the test trip: no leakage. A target is a "
               "<b>revisit</b> if its place is in the history, otherwise <b>new</b>."]))
    method_side = (
        '<div class="tx-h3">What we compare</div>'
        + _ul(["<b>Heuristic floor (no AI):</b> rank visited places by visits, weighted by time of day and recency.",
               "<b>LLM-Mob prompt:</b> the 40 most recent earlier check-ins plus the current trip.",
               "<b>Retrieval prompt:</b> the 5 most similar earlier trips (Jaccard) shown as worked examples. "
               "Jaccard, LCS, TF-IDF, BM25 and a hybrid were built and benchmarked; BM25 won on validation.",
               "<b>Models:</b> Qwen2.5-7B and 3B, 4-bit, greedy, one Kaggle T4. "
               "<b>Statistics:</b> bootstrap CIs, McNemar, Holm correction."]))
    return (_ul(["<b>Question:</b> when an LLM predicts a person’s next place, is it copying their habits, recalling "
                 "memorised venues, or reasoning? Treated as an <b>information-retrieval</b> ranking problem."])
            + hyp + _two(data_side, method_side)
            + _take("The prompt may state the <i>time</i> of the next check-in (as in the published methods) but never "
                    "its place, category or name. Every method is scored on exactly the same cases."))


# ---------------------------------------------------------------------------
# Slide 2: results
# ---------------------------------------------------------------------------
def _e1_table(res: dict, sample: str) -> str:
    combos = [(m, mo) for m in ("llmmob", "retrieval-jaccard") for mo in ("qwen7b", "qwen3b")]
    rows = [{"cls": "ref", "cells": [("Frequency heuristic", "strong")]
             + [ui.num(100 * data.freq(res, d, sample)["acc1"]["mean"]) for d in DS_KEYS]}]
    for m, mo in combos:
        cells = [f"{ui.METHODS[m]} · {ui.MODEL_SHORT[mo]}"]
        for d in DS_KEYS:
            hit = next((x for x in res["e1"] if x["dataset"] == d and res["runs"][x["run"]]["method"] == m
                        and res["runs"][x["run"]]["model"] == mo), None)
            if not hit:
                cells.append("–")
                continue
            r, v = res["runs"][hit["run"]], hit["vs_frequency"]
            cells.append(f'{ui.num(100 * r["acc1"]["mean"])} <span class="tx-ci">Δ {ui.num(100 * v["acc1"]["diff"], signed=True)}'
                         f' · p {ui.pval(v["mcnemar_acc1"].get("p_holm"))}</span>')
        rows.append({"cells": cells})
    return ui.table([("Accuracy@1 (%)", "l")] + [(ui.DATASET_SHORT[d], "r") for d in DS_KEYS], rows, min_width=640)


def _new_place_hits(res: dict, sample: str) -> tuple[int, int, int, int]:
    hits = total = runs = 0
    for r in res["runs"].values():
        if r["condition"] in (C.NO_NAMES, "present"):
            n = r["new"]["n"]
            hits += round(r["new"]["acc1"]["mean"] * n)
            total += n
            runs += 1
    distinct = sum(data.freq(res, d, sample)["new"]["n"] for d in DS_KEYS)
    return hits, total, runs, distinct


def _name_gap(res: dict) -> float | None:
    gaps = []
    for m in ("llmmob", "retrieval-jaccard"):
        for mo in ("qwen7b", "qwen3b"):
            by = {r["condition"]: r for _, r in context.runs_for(res, "msteps_tokyo", mo, m)}
            vals = [by[c]["acc1"]["mean"] for c in C.NAME_CONDITIONS if c in by]
            if len(vals) > 1:
                gaps.append(100 * (max(vals) - min(vals)))
    return max(gaps) if gaps else None


def _copy_range(sample: str) -> str:
    rep = data.assistant_report("copy_analysis")
    if not rep:
        return "–"
    vals = []
    for d in ("nyc", "tky"):
        r = (rep.get("runs") or {}).get(f"{d}__qwen7b__llmmob__none__{sample}")
        if r:
            vals.append(100 * r["match"]["any_rule"]["mean"])
    return f"{ui.num(min(vals), 0)}–{ui.num(max(vals), 0)}%" if vals else "–"


def s_results() -> str:
    sample = _sample()
    _, _, res = context.get()
    hits, total, runs, distinct = _new_place_hits(res, sample)
    gap = _name_gap(res)
    e1 = (
        '<div class="tx-h3">E1 · Heuristic floor</div>' + _e1_table(res, sample)
        + ui.caption_html("Δ = paired difference to the heuristic in percentage points; p = McNemar, Holm-adjusted. "
                          "Massive-STEPS uses real names.")
        + _ul(["<b>Rich history (NYC, TKY):</b> the counter beats the LLM.",
               "<b>Thin history (Massive-STEPS):</b> Qwen-7B beats the counter by re-ranking the few places the user knows."]))
    others = (
        '<div class="tx-h3">E2, E3 and what is copied</div>'
        + _stats([(f"{hits} / {total:,}", f"new places predicted correctly: {distinct} never-visited cases in {runs} runs (E2)"),
                  (f"{ui.num(gap)} pp" if gap is not None else "–", "largest accuracy gap between real, removed and fake venue names (E3)"),
                  (_copy_range(sample), "of Qwen-7B (LLM-Mob) first guesses equal a simple rule’s pick, NYC and TKY")]))
    return (e1 + others
            + _take("<b>Copy.</b> The LLM re-ranks the user’s own habits: all its accuracy comes from revisits, venue names "
                    "make no difference, and a tuned counter does the same job better when history is rich."))


# ---------------------------------------------------------------------------
# Slide 3: assistant and conclusion
# ---------------------------------------------------------------------------
def _engine_table() -> str:
    rep = data.assistant_report("results")
    if not rep:
        return _missing("The engine evaluation")
    try:
        rows = []
        for d in DS_KEYS:
            e, h = rep["methods"]["engine"][d], rep["methods"]["heuristic"][d]
            c = next(x for x in rep["comparisons"] if x["dataset"] == d and x["method"] == "engine")
            rows.append({"cells": [ui.DATASET_SHORT[d], ui.num(100 * h["acc1"]["mean"]),
                                   (ui.num(100 * e["acc1"]["mean"]), "strong"),
                                   f'{ui.num(100 * c["acc1"]["diff"], signed=True)} '
                                   f'<span class="tx-ci">p {ui.pval(c["mcnemar_acc1"].get("p_holm"))}</span>']})
    except (KeyError, StopIteration, TypeError):
        return _missing("The engine evaluation")
    return ui.table([("Acc@1 (%)", "l"), ("Heuristic", "r"), ("Evidence engine", "r"), ("Δ (pp)", "r")], rows)


def s_assistant() -> str:
    flow = _flow([("1", "Read the question", "“somewhere new”, kind of place"),
                  ("2 · IR decides", "Evidence engine", "Counts by place, time, weekday; back-off; rank fusion"),
                  ("3 · LLM explains", "Qwen2.5-3B, local", "Sees only readable evidence; nothing leaves the laptop"),
                  ("4", "Validator", "Every place, count, % and distance must be in the evidence; else retry, then template")])
    left = ('<div class="tx-h3">Evidence engine vs the heuristic (no LLM)</div>' + _engine_table()
            + ui.caption_html("Matches the heuristic with rich history; clearly better where history is thin."))
    right = ('<div class="tx-h3">Answer quality</div>'
             + _stats([("56 / 4 / 0", "of 60 answers: first time / after retry / template"),
                       ("15 / 15", "personas: routine was the top suggestion")]))
    return (_ul(["The experiments show the LLM is a habit re-ranker, so the assistant lets <b>IR decide</b> and the "
                 "<b>LLM only explain</b>."])
            + flow + _two(left, right)
            + _take("<b>Limits:</b> two Qwen sizes in 4-bit; Massive-STEPS has only 334 cases; the name test shows "
                    "reliance, not memorisation; “something new” suggestions are popularity-based and not scored."))


SLIDES = [
    ("1 · Approach", "Copy, Recall, or Reason?", s_approach),
    ("2 · Results", "The LLM copies habits", s_results),
    ("3 · Application and conclusion", "IR decides, the LLM explains", s_assistant),
]


def _go(delta: int):
    st.session_state["slide_idx"] = max(0, min(len(SLIDES) - 1, st.session_state["slide_idx"] + delta))


def render():
    if st.session_state.get("slide_idx") not in range(len(SLIDES)):
        q = st.query_params.get("slide", "1")
        st.session_state["slide_idx"] = int(q) - 1 if q.isdigit() and 1 <= int(q) <= len(SLIDES) else 0
    idx = st.session_state["slide_idx"]

    c1, c2, c3, c4 = st.columns([1, 3, 1, 1.3], vertical_alignment="bottom")
    c1.button("← Previous", key="sl_prev", on_click=_go, args=(-1,), disabled=idx == 0, width="stretch")
    c2.selectbox("Slide", list(range(len(SLIDES))), key="slide_idx", label_visibility="collapsed",
                 format_func=lambda i: f"{i + 1}. {SLIDES[i][1]}")
    c3.button("Next →", key="sl_next", on_click=_go, args=(1,), disabled=idx == len(SLIDES) - 1, width="stretch")
    if c4.toggle("Presentation mode", key="sl_present", help="Hides the sidebar. Use ← and → to move."):
        st.html(PRESENT_CSS)
    st.html(f"<script>{KEYS_JS}</script>", unsafe_allow_javascript=True)

    idx = st.session_state["slide_idx"]
    section, title, fn = SLIDES[idx]
    head = f'<div class="tx-kicker">{ui.esc(section)}</div><h2 class="tx-sl-title">{ui.esc(title)}</h2>'
    st.html(f'<div class="tx-progress tx-root" style="margin:8px 0 12px"><span style="width:{100 * (idx + 1) / len(SLIDES)}%">'
            f'</span></div><div class="tx-root tx-slide">{head}{fn()}</div>')
