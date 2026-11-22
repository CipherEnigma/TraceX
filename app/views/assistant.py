"""Page 7: Personal assistant - IR evidence decides where a user may go next, Qwen2.5-3B only explains it.

Role-play a real user from the data: pick the moment (day, time, current place), ask in plain words, and see
the answer next to the exact evidence it was built from (docs/assistant_plan.md, task 3.5-3.6).
"""
from __future__ import annotations

import math
import re

import pydeck as pdk
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


ROLE = {"here": ("You are here", MAP_COLORS["here"]), "suggested": ("Suggested next", MAP_COLORS["usual"]),
        "usual": ("Usual place", MAP_COLORS["usual"]), "new": ("New place nearby", MAP_COLORS["new"])}
ROLE_MARK = {"here": "● ", "suggested": "★ ", "new": "+ ", "usual": ""}


def _rgb(hex_color: str, a: int = 255) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def _km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _short(text: str, n: int = 34) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _map_points(prof, ev, places, current: int | None) -> dict[int, str]:
    """poi_id -> role, in chip order: current place, the last answer's suggestions, new places, usual places."""
    pts: dict[int, str] = {}
    if current is not None:
        pts[current] = "here"
    for s_ in (ev.habits[:3] if ev else []):
        pts.setdefault(s_.poi_id, "suggested")
    for e in (ev.explore if ev else []):
        pts.setdefault(e.poi_id, "new")
    for pid, _ in prof.top_places[:10]:
        pts.setdefault(pid, "usual")
    return pts


def _place_card(pid: int, role: str, prof, ev, places, current: int | None) -> str:
    p = places.get(pid)
    visits = dict(prof.top_places).get(pid, 0)
    title, color = ROLE[role]
    rows = [("Name", ui.esc(p.name))] if p.name else []
    rows.append(("Kind of place", ui.esc(p.category)))
    rows.append(("Visits by this user", f"{visits} of {prof.n_checkins}" if visits else "never visited"))
    if current is not None and pid != current:
        rows.append(("Distance from here", f"{_km(places.get(current), p):.1f} km"))
    rows.append(("Check-ins by everyone", f"{p.popularity:,}"))
    why = next((s_.facts[0] for s_ in (ev.habits if ev else []) if s_.poi_id == pid and s_.facts), None)
    if why:
        rows.append(("Why it is suggested", f"{why.count} of {why.total} ({ui.num(100 * why.share, 0)}%) · "
                                            f"{ui.esc(re.sub(r'^\s*you,\s*', '', why.level))}"))
    body = "".join(f'<div class="kv"><span class="tx-note">{k}</span><span>{v}</span></div>' for k, v in rows)
    return (f'<div class="tx-tile tx-root tx-placecard"><span class="lbl"><span class="sw sw-lg" '
            f'style="background:{color}"></span>{title}</span>'
            f'<div class="tx-h3">{ui.esc(llm.label(places, pid))}</div>{body}</div>')


def _map(prof, ev, places, current: int | None, key: str):
    """Pick a place in the chips or click it on the map: it is ringed, labelled, zoomed to and described."""
    pts = _map_points(prof, ev, places, current)
    hl_key, map_key = f"{key}_hl", f"{key}_map"
    if st.session_state.get(hl_key) not in pts:
        st.session_state[hl_key] = None

    def picked_on_map():  # a click on the map sets the same state as the chips, before the rerun
        state = st.session_state.get(map_key) or {}
        objs = (state.get("selection") or {}).get("objects", {}).get("places", [])
        if objs:
            st.session_state[hl_key] = int(objs[0]["poi_id"])

    st.html(ui.legend([(ui.sw("var(--s3)"), "You are here"), (ui.sw("sw-7"), "Usual places and suggestions"),
                       (ui.sw("sw-a"), "New places nearby"), ('<span class="lg-ring"></span>', "Selected")]))
    sel = st.pills("Highlight a place (or click a point on the map)", list(pts), key=hl_key,
                   format_func=lambda pid: ROLE_MARK[pts[pid]] + _short(llm.label(places, pid)))
    rows = []
    for pid, role in pts.items():
        p = places.get(pid)
        rows.append({"poi_id": pid, "lat": p.lat, "lon": p.lon, "label": llm.label(places, pid),
                     "role": ROLE[role][0], "color": _rgb(ROLE[role][1], 235),
                     "r": 9 if role == "here" else 7 if role in ("suggested", "new") else 6})
    # pydeck reads plain strings as JS expressions, so literal strings are written '"like this"'
    rows.sort(key=lambda r: r["role"] == ROLE["here"][0])  # "you are here" drawn last, i.e. on top
    layers = [pdk.Layer("ScatterplotLayer", id="places", data=rows, get_position="[lon, lat]",
                        get_fill_color="color", get_radius="r", radius_units='"pixels"', stroked=True,
                        get_line_color=[255, 255, 255, 230], line_width_min_pixels=1.5, pickable=True,
                        auto_highlight=True, highlight_color=[31, 78, 140, 90])]
    if sel is not None:
        hp = next(r for r in rows if r["poi_id"] == sel)
        layers += [
            pdk.Layer("ScatterplotLayer", id="ring", data=[hp], get_position="[lon, lat]", get_radius=18,
                      radius_units='"pixels"', filled=True, get_fill_color=[42, 120, 214, 45], stroked=True,
                      get_line_color=[42, 120, 214, 255], line_width_min_pixels=3),
            pdk.Layer("TextLayer", id="label", data=[hp], get_position="[lon, lat]", get_text="label",
                      get_size=13, get_color=[11, 11, 11, 255], get_pixel_offset=[0, -32], character_set='"auto"',
                      font_family='"IBM Plex Sans, IBM Plex Sans JP, Noto Sans JP, sans-serif"',
                      background=True, get_background_color=[252, 252, 251, 240], background_padding=[6, 3]),
        ]
        view = pdk.ViewState(latitude=hp["lat"], longitude=hp["lon"], zoom=15)
    else:
        view = pdk.data_utils.compute_view([[r["lon"], r["lat"]] for r in rows], view_proportion=0.95)
        view.zoom = min(view.zoom, 15)
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style=None,
                    tooltip={"html": "<b>{label}</b><br/>{role}",
                             "style": {"fontFamily": "IBM Plex Sans, sans-serif", "fontSize": "12px"}})
    mcol, ccol = st.columns([2.2, 1], gap="medium")
    with mcol:
        st.pydeck_chart(deck, height=380, on_select=picked_on_map, selection_mode="single-object", key=map_key)
    with ccol:
        if sel is None:
            st.html('<div class="tx-tile tx-root tx-placecard"><span class="lbl">Place details</span>'
                    '<span class="tx-note">Pick a place above or click a point on the map to highlight it and see '
                    'its details here.</span></div>')
        else:
            st.html(_place_card(sel, pts[sel], prof, ev, places, current))


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
                f'across all days · example moment: {WD_LONG[persona["weekday"]]} {persona["hour"]:02d}:00'
                f'</span></div>')
    with b:
        st.button("Load this routine", key=f"{here_key}_load", on_click=load, width="stretch",
                  help="Set the current place to the routine's start and the day and time to one example moment when "
                       "it happened, then ask “Where should I go next?”. The answer may quote narrower counts for "
                       "that day and time than the all-days count here.")


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

        ui.section("Map", "pick a place or click a point to highlight it")
        last_ev = history[-1]["ev"] if history else None
        _map(prof, last_ev, places, current, chat_key)
