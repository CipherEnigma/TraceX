"""Page 7: Personal assistant - IR evidence decides where a user may go next, Qwen2.5-3B only explains it.

Role-play a real user from the data: pick the moment (day, time, current place), ask in plain words, and see
the answer next to the exact evidence it was built from (docs/assistant_plan.md, task 3.5-3.6).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import ui
from tracex_assist import api, llm
from tracex_assist.contract import TIME_BUCKETS, WEEKDAYS, Context, routine_key, time_bucket

DATASETS = {"nyc": "NYC", "tky": "TKY", "msteps_tokyo": "Massive-STEPS Tokyo"}
WD_LONG = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SUGGESTED = ["Where should I go next?", "Somewhere new to eat?", "Where do I usually get coffee?"]
KIND_LABEL = {"transition": "Came next after this place", "frequency": "Share of all your check-ins",
              "time_habit": "Your visits at this time", "similar_trip": "Similar past trips",
              "category": "Kind of place next"}
MAP_COLORS = {"here": "#EB6834", "usual": "#2A78D6", "new": "#1BAF7A"}  # = --s3, --s7, --sa


# ---------------------------------------------------------------------------
# Cached data access (everything goes through tracex_assist.api: real modules or the stub)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _places(dataset: str):
    return api.place_index(dataset)


@st.cache_data(show_spinner=False)
def _personas(dataset: str) -> list[dict]:
    return api.demo_users(dataset, 12)


@st.cache_resource(show_spinner=False)
def _profile(dataset: str, user_id: str):
    return api.build_profile(dataset, user_id)


@st.cache_data(ttl=15, show_spinner=False)
def _ollama() -> dict:
    return llm.ollama_status(llm.DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------
def _status_tag(s: dict) -> str:
    if s["running"] and s["has_model"]:
        return ui.tag("● Qwen2.5-3B ready", "ok")
    if s["running"]:
        return ui.tag(f"Ollama on, {llm.DEFAULT_MODEL} missing: template answers", "bad")
    return ui.tag("Ollama off: template answers")


def _profile_card(prof, places) -> str:
    top = "".join(
        f'<li><span class="rk">{i}</span><span class="cat" title="{ui.esc(llm.label(places, pid))}">'
        f'{ui.esc(llm.label(places, pid))}</span><span class="tx-ci">{n}</span></li>'
        for i, (pid, n) in enumerate(prof.top_places[:5], 1))
    return (f'<div class="tx-tile tx-root" style="gap:12px"><span class="lbl">Profile · user {ui.esc(prof.user_id)}</span>'
            f'<div class="tx-row" style="justify-content:flex-start;gap:20px;flex-wrap:wrap">'
            f'<span><span class="tx-big">{prof.n_checkins:,}</span> <span class="tx-ci">check-ins</span></span>'
            f'<span><span class="tx-big">{prof.n_trips:,}</span> <span class="tx-ci">trips</span></span></div>'
            f'<span class="tx-note">{ui.esc(prof.first_day)} to {ui.esc(prof.last_day)}</span>'
            f'<div class="tx-h3">Usual places</div><ol class="tx-as-list">{top}</ol></div>')


def _routine(prof, weekday: int, hour: int) -> str:
    buckets = list(TIME_BUCKETS)
    counts = {k: sum(v.values()) for k, v in prof.routine.items()}
    vmax = max(counts.values(), default=1)
    now = routine_key(weekday, hour)
    head = '<span></span>' + "".join(f'<span class="h">{b[:3]}</span>' for b in buckets)
    rows = []
    for d, wd in enumerate(WEEKDAYS):
        cells = []
        for b in buckets:
            k = f"{wd}|{b}"
            n = counts.get(k, 0)
            top = max(prof.routine.get(k, {}).items(), key=lambda x: x[1], default=None)
            tip = f"{WD_LONG[d]} {b}: {n} check-ins" + (f", mostly {top[0]} ({top[1]})" if top else "")
            cells.append(f'<span class="c{" now" if k == now else ""}" title="{ui.esc(tip)}" '
                         f'style="--a:{round(100 * n / vmax) if n else 0}%"></span>')
        rows.append(f'<span class="d">{wd}</span>' + "".join(cells))
    return (f'<div class="tx-tile tx-root" style="gap:8px"><div class="tx-row"><span class="lbl">Weekly routine</span>'
            f'<span class="tx-note">darker = more · outlined = now</span></div>'
            f'<div class="tx-heat" role="img" aria-label="Check-ins by weekday and time of day">{head}{"".join(rows)}</div>'
            f'</div>')


def _share_bar(share: float) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:8px;justify-content:flex-end">'
            f'<span style="min-width:36px">{ui.num(100 * share, 0)}%</span><span class="tx-prog">'
            f'<span style="width:{100 * share:.0f}%;background:var(--accent)"></span></span></span>')


def _evidence_html(ev, places) -> str:
    rows = [{"cells": [(f'{KIND_LABEL.get(f.kind, f.kind)}<span class="tx-note">{ui.esc(f.level)}</span>', "two"),
                       (ui.esc(llm.label(places, f.poi_id)) if f.poi_id is not None else ui.esc(f.category or "–"),
                        "wrap"),
                       (f"{f.count} of {f.total}", "r"), (_share_bar(f.share), "r")]}
            for f in ev.facts]
    parts = [ui.table([("Evidence · based on", "l"), ("Place", "l"), ("Count", "r"), ("Share", "r")], rows)
             if rows else '<p class="tx-note">No past visits match this moment.</p>']
    if ev.explore:
        new_rows = [{"cells": [(ui.esc(llm.label(places, e.poi_id)), "wrap"), (f"{e.km:.1f} km", "r"),
                               (f"{e.popularity:,}", "r")]} for e in ev.explore]
        parts.append('<div class="tx-h3" style="margin-top:12px">Places you have never visited, nearby</div>' +
                     ui.table([("Place", "l"), ("Distance", "r"), ("Check-ins by others", "r")], new_rows))
    notes = []
    if ev.thin:
        notes.append(ui.tag("thin evidence: fewer than 5 past visits", "bad"))
    acc = (ev.confidence or {}).get("acc1")
    notes.append(ui.tag(f"offline top-1 accuracy {ui.num(100 * acc)}% (n = {ev.confidence.get('n')})") if acc else
                 ui.tag("offline accuracy: not measured yet"))
    return '<div class="tx-root" style="display:flex;flex-direction:column;gap:8px">' + "".join(parts) + \
        f'<div style="display:flex;gap:8px;flex-wrap:wrap">{"".join(notes)}</div></div>'


def _meta_line(res) -> str:
    src = {"llm": "Qwen2.5-3B", "llm-retry": "Qwen2.5-3B · after one stricter retry",
           "template": "template answer (built from the evidence, no LLM)"}[res.source]
    return f'<div class="tx-note">{src} · {res.seconds:.1f} s · checked: every place and number comes from the evidence</div>'


def _map(prof, ev, places, current: int | None):
    pts = []
    for pid, _ in prof.top_places[:10]:
        p = places.get(pid)
        pts.append((p.lat, p.lon, MAP_COLORS["usual"], 60))
    for e in (ev.explore if ev else []):
        p = places.get(e.poi_id)
        pts.append((p.lat, p.lon, MAP_COLORS["new"], 60))
    if current is not None:
        p = places.get(current)
        pts.append((p.lat, p.lon, MAP_COLORS["here"], 90))
    st.html(ui.legend([(ui.sw("var(--s3)"), "You are here"), (ui.sw("sw-7"), "Your usual places"),
                       (ui.sw("sw-a"), "New places suggested")]))
    st.map(pd.DataFrame(pts, columns=["lat", "lon", "color", "size"]), latitude="lat", longitude="lon",
           color="color", size="size", height=360)  # auto-fits all points


def _routine_callout(persona: dict, places, here_key: str):
    """Member 1's demo persona: one strong, real routine of this user, loadable in one click."""
    def load():
        st.session_state["as_day"] = persona["weekday"]
        st.session_state["as_hour"] = persona["hour"]
        st.session_state[here_key] = persona["from_poi"]

    share = persona["count"] / persona["total"] if persona["total"] else 0
    a, b = st.columns([5, 1], vertical_alignment="center")
    with a:
        st.html(f'<div class="tx-callout tx-root"><span class="tx-tag ok">★ Known routine</span><span>After '
                f'<b>{ui.esc(llm.label(places, persona["from_poi"]))}</b> this user went to '
                f'<b>{ui.esc(llm.label(places, persona["to_poi"]))}</b> {persona["count"]} of {persona["total"]} times '
                f'({ui.num(100 * share, 0)}%), usually about {ui.num(persona["median_gap_hours"], 1)} h later · '
                f'typically {WD_LONG[persona["weekday"]]} around {persona["hour"]:02d}:00</span></div>')
    with b:
        st.button("Load this routine", key=f"{here_key}_load", on_click=load, width="stretch",
                  help="Set the day, time and current place to this routine, then ask “Where should I go next?”")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render():
    ui.page_header("Personal assistant", "Role-play a real user from the data and ask where to go next. The evidence "
                                         "engine decides from their history; Qwen2.5-3B only puts it into words.")
    status = _ollama()
    c1, c2, c3 = st.columns([1.3, 1.6, 1.3], vertical_alignment="bottom")
    with c1:
        ds = ui.segmented("Dataset", DATASETS, "as_ds")
    personas = _personas(ds)
    with c2:
        uid = ui.select("User", {p["user_id"]: ("★ " if p.get("persona") else "") + f"User {p['user_id']} · "
                                               f"{p['n_checkins']:,} check-ins · mostly {p['top_category']}"
                                 for p in personas}, f"as_user_{ds}")
    with c3:
        st.html(f'<div class="tx-root" style="text-align:right;padding-bottom:10px">{_status_tag(status)}</div>')
    ui.rule()
    if any(api.USING_STUB.values()):
        st.html('<div class="tx-callout tx-root"><span class="tx-tag">Preview</span><span class="tx-note">Evidence comes '
                'from a simple stand-in until the profile (Member 1) and evidence engine (Member 2) modules are merged. '
                'Numbers are real counts from the data but not the final engine.</span></div>')

    places = _places(ds)
    prof = _profile(ds, uid)
    chat_key = f"as_chat_{ds}_{uid}"
    st.session_state.setdefault(chat_key, [])
    persona = next((p.get("persona") for p in personas if p["user_id"] == uid), None)
    here_key = f"as_here_{ds}_{uid}"
    if persona:
        _routine_callout(persona, places, here_key)

    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.html('<div class="tx-root tx-h2">The moment</div>')
        d1, d2 = st.columns(2)
        # defaults live in session state only, so "Load this routine" can set them without a widget warning
        st.session_state.setdefault("as_day", 4)
        st.session_state.setdefault("as_hour", 18)
        with d1:
            weekday = st.selectbox("Day", list(range(7)), format_func=lambda d: WD_LONG[d], key="as_day")
        with d2:
            hour = st.selectbox("Time", list(range(24)), format_func=lambda h: f"{h:02d}:00", key="as_hour")
        visits = dict(prof.top_places)
        here_ids = [pid for pid, _ in prof.top_places[:15]]
        if persona and persona["from_poi"] not in here_ids:  # the routine's start place is always selectable
            here_ids.append(persona["from_poi"])
        here_opts = {pid: f"{llm.label(places, pid)} · {visits.get(pid, 0)} visits" for pid in here_ids}
        if st.session_state.get(here_key) not in here_opts:
            st.session_state[here_key] = here_ids[0]
        current = st.selectbox("You are at", here_ids, format_func=lambda pid: here_opts[pid], key=here_key)
        want_new = st.toggle("Suggest something new", key="as_new",
                             help="Also list nearby places this user has never visited")
        st.html(f'<div class="tx-note">{WD_LONG[weekday]} {hour:02d}:00 · {time_bucket(hour)}</div>')
        st.html(_profile_card(prof, places))
        st.html(_routine(prof, weekday, hour))

    with right:
        st.html('<div class="tx-root tx-sechead" style="margin-top:0"><h2 class="tx-h2">Ask</h2>'
                '<span class="tx-note">answers use only this user’s history</span></div>')
        chips = st.columns(len(SUGGESTED))
        clicked = None
        for col, q in zip(chips, SUGGESTED):
            if col.button(q, key=f"as_q_{q}", width="stretch"):
                clicked = q
        typed = st.chat_input("Ask about your next stop, e.g. “I just left the gym, where should I go?”",
                              key="as_input")
        question = typed or clicked
        if question:
            intent = llm.parse_intent(question)
            ctx = Context(weekday=weekday, hour=hour, current_poi=current, want_new=want_new or intent["want_new"],
                          want_kind=intent["want_kind"])
            with st.spinner("Gathering evidence and writing the answer…"):
                ev = api.gather(prof, ctx, places)
                res = llm.answer(question, ev, places, use_llm=status["running"] and status["has_model"])
            st.session_state[chat_key].append({"q": question, "res": res, "ev": ev})

        history = st.session_state[chat_key]
        if not history:
            ui.empty_state("Ask a question to start", "Set the day, time and current place on the left, then type a "
                           "question or pick one above. Every answer comes with the evidence it was built from.")
        for i, turn in enumerate(history):
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(turn["q"])
            with st.chat_message("assistant", avatar=":material/explore:"):
                st.markdown(turn["res"].text)
                st.html(_meta_line(turn["res"]))
                with st.expander("How I got this", expanded=(i == len(history) - 1)):
                    st.html(_evidence_html(turn["ev"], places))
                    st.caption("Exactly what the model was shown")
                    st.json(turn["res"].view, expanded=False)
        if history:
            if st.button("Clear conversation", key=f"as_clear_{chat_key}", icon=":material/delete:"):
                st.session_state[chat_key] = []
                st.rerun()

        ui.section("Map", "usual places of this user, the current place and suggested new places")
        last_ev = history[-1]["ev"] if history else None
        _map(prof, last_ev, places, current)
