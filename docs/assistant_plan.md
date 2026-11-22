# TraceX Personal Assistant: execution plan (5 hours, 3 members)

## 1. What we are adding and why

Our experiments showed that LLMs predict the next place by **copying habits**: no better than a visit counter, 0% on
never-visited places, no use of venue names. So the final piece of the project uses each tool for what it is good at:

- **IR and statistics decide** where the user is likely to go and why (exact counts, percentages, nearby places).
- **The LLM only explains** that evidence in friendly language. It never invents a place or a number.

**The demo.** In the dashboard you pick a real user from the dataset (NYC or Massive-STEPS Tokyo), set the day, time and
current place, and ask in plain words, e.g. *"I just left the gym, where should I go?"* The assistant answers:

> After the gym you usually go to **Fast Food Restaurant #812**: 11 of your last 20 visits (55%). Next most likely is
> **Coffee Shop #345** (4 of 20). Want something new? Three fast-food places within 1 km you have never visited:
> #1201 (0.3 km, 140 check-ins by others), #998 (0.6 km), #2210 (0.9 km).
> *Based on 20 past gym visits. My top suggestion matches the real next stop about 1 time in 4.*

with a **"How I got this"** panel (the actual numbers) and a **map**. It runs locally with Ollama (`qwen2.5:3b`) and
falls back to a template answer when Ollama is not installed, so the dashboard works on every laptop.

**What it adds to the paper:** stronger non-LLM baselines (Markov, recency, fusion), an honest accuracy for the
assistant, a behaviour analysis of what the LLM copies, and a design lesson: *use LLMs to explain IR evidence, not to
predict*.

**Data limits to keep in mind:** NYC has categories and coordinates but **no venue names** (places are shown as
"Fast Food Restaurant #812"). Tokyo has real names and addresses but fewer check-ins per user. "Being a user" means
role-playing one of the dataset's users.

---

## 2. Architecture and ownership

```
                 Member 1                      Member 2                        Member 3 (Aayush)
            tracex_assist/profile.py      tracex_assist/evidence.py       tracex_assist/llm.py + app/
            user profile, routine,   ->   habits, back-off, fusion,  ->   grounded LLM answer, validator,
            place index, nearby,          category prediction,            fallback, chat page, map,
            demo users                    explore, confidence             evidence panel
                     \                          |                               /
                      \--------- tracex_assist/contract.py (shared contract) -----/
            tracex_assist/analysis_copy.py   tracex_assist/evaluate.py      tracex_llm (no change)
            what does the LLM copy?          offline accuracy + LLM fusion
```

Each member owns separate files, so the three branches merge without conflicts. **Nobody edits another member's
files**; ask the owner instead. `contract.py` is written first (step 0) and only changed by agreement.

---

## 3. The shared contract: `tracex_assist/contract.py` (step 0, everyone, 20 min)

```python
from dataclasses import dataclass, field

TIME_BUCKETS = {"morning": (5, 11), "midday": (11, 14), "afternoon": (14, 17), "evening": (17, 21), "night": (21, 5)}

@dataclass
class Place:            # one POI
    poi_id: int
    category: str
    name: str | None    # None on NYC
    lat: float
    lon: float
    popularity: int     # check-ins by all users in train

@dataclass
class Visit:            # one check-in of the user
    poi_id: int
    time: str           # "2013-01-04 18:05"
    weekday: int        # 0 = Monday
    hour: int
    trip: str           # trajectory id
    pos: int            # position inside the trip

@dataclass
class Profile:          # Member 1 builds it
    dataset: str
    user_id: str
    trips: list[list[Visit]]                  # oldest first
    top_places: list[tuple[int, int]]         # (poi_id, visits), most visited first
    routine: dict[str, dict[str, int]]        # "Fri|evening" -> {category: visits}
    n_checkins: int
    n_trips: int
    first_day: str
    last_day: str
    home: tuple[float, float]                 # centroid of the user's places

@dataclass
class Context:          # the question's situation (from the UI pickers)
    weekday: int
    hour: int
    current_poi: int | None                   # where the user is now
    want_new: bool = False                    # "suggest something new"

@dataclass
class Fact:             # one piece of evidence, always with its counts
    kind: str           # "transition" | "time_habit" | "frequency" | "similar_trip" | "category"
    poi_id: int | None
    category: str | None
    count: int          # e.g. 11
    total: int          # e.g. 20 -> share 0.55
    level: str          # back-off level, e.g. "you, after this place, on Fridays"

@dataclass
class Suggestion:
    poi_id: int
    score: float
    facts: list[Fact]

@dataclass
class Explore:
    poi_id: int
    km: float
    popularity: int
    category: str

@dataclass
class Evidence:         # Member 2 returns it, Member 3 explains and displays it
    context: Context
    habits: list[Suggestion]                  # top 5 known places
    categories: list[tuple[str, float]]       # top 3 categories with share
    explore: list[Explore]                    # up to 5 new places
    facts: list[Fact]                         # everything shown in "How I got this"
    confidence: dict                          # {"acc1": 0.25, "acc5": 0.59, "n": 500} from offline evaluation
    thin: bool                                # True if the evidence rests on fewer than 5 visits
```

**Function signatures (the only cross-member calls):**

| Owner | Function |
|---|---|
| M1 | `place_index(dataset) -> PlaceIndex` with `.get(poi_id) -> Place` and `.nearby(lat, lon, km, category=None) -> list[tuple[Place, float]]` |
| M1 | `build_profile(dataset, user_id, history=None) -> Profile` (`history` = list of trajectories; default = all of the user's train trips) |
| M1 | `demo_users(dataset, n=20) -> list[dict]` with `user_id, n_checkins, n_trips, top_category` |
| M2 | `gather(profile, context, places) -> Evidence` |
| M2 | `evidence_to_dict(ev) -> dict` (JSON-safe; used by the LLM prompt and the UI) |
| M3 | `answer(question, ev, model="qwen2.5:3b") -> Iterator[str]`, `fallback_answer(ev) -> str`, `ollama_status() -> dict` |

**Leakage rule (needed for the honest offline accuracy):** `build_profile(..., history=ds.history(inst))` must use only
the trips given; `gather` may use the context (target weekday and hour, last prefix place) but never the target place.
This is the same rule as the LLM experiments.

---

## 4. Tasks per member

Each member has about 3.5 hours of own work plus shared steps. Times are targets, not limits.

### Member 1: data, profiles and behaviour analysis

| # | Task | Output | Time |
|---|---|---|---|
| 1.1 | `place_index`: load POIs, add train popularity, fast `nearby()` (haversine on a lat/lon grid) | `tracex_assist/profile.py` | 45 min |
| 1.2 | `build_profile`: trips → visits, top places, routine by weekday × time bucket, home centroid; respects `history` | `profile.py` | 60 min |
| 1.3 | `demo_users`: rank users by richness (check-ins, trips, distinct places); pick **5 demo personas** per dataset with a clear routine (e.g. gym → food); write them to `data/assistant/demo_users.json` | json + function | 30 min |
| 1.4 | **What does the LLM copy?** For every saved LLM run: share of top-1 guesses equal to the frequency top-1, the last place visited, the Markov top-1, or a place from the current trip; and where the LLM beats the counter (by history length, time of day, revisit/new) | `tracex_assist/analysis_copy.py` → `reports/assistant/copy_analysis.{json,md}` | 60 min |
| 1.5 | Tests: profile on a synthetic user, `history` respected, `nearby` correct, time buckets wrap midnight | `tests/test_assist_profile.py` | 20 min |

**Done when:** `build_profile("nyc", "<demo user>")` returns in under 1 s; tests pass; the copy-analysis table exists.

### Member 2: evidence engine and offline evaluation

| # | Task | Output | Time |
|---|---|---|---|
| 2.1 | Transition statistics with **back-off**: (you, this place, this weekday/time bucket) → (you, this place, any time) → (you, this category) → (everyone, this category). Record the level and counts used | `tracex_assist/evidence.py` | 60 min |
| 2.2 | Habit ranking: fuse the time-aware heuristic (`frequency_baseline`, merged in PR #4), Markov (last place → next) and time-habit rankings with **reciprocal rank fusion (k = 60)**; each suggestion keeps its facts | `evidence.py` | 40 min |
| 2.3 | **Category prediction** (top 3 next categories with shares) and **explore** list: same category as the top predicted category, within 1 km of the current place (widen to 2 / 5 km if empty), never visited by the user, ranked by popularity | `evidence.py` | 40 min |
| 2.4 | `confidence` and `thin` flag; `evidence_to_dict` | `evidence.py` | 15 min |
| 2.5 | **Offline evaluation** on `sample_test_500` (NYC, TKY, Massive-STEPS) with the leakage rule: Acc@1/5, MRR, NDCG@5 for frequency, Markov, recency, the fused engine, plus **category-level Acc@1/5**; paired bootstrap + McNemar vs frequency, Holm-corrected. Also the post-hoc **LLM fusion** (saved LLM ranking fused with the engine), clearly labelled exploratory | `tracex_assist/evaluate.py` → `reports/assistant/results.{json,md}` | 60 min |
| 2.6 | Tests: back-off picks the right level, shares add up, explore never returns a visited place, no target leakage | `tests/test_assist_evidence.py` | 20 min |

**Done when:** `gather()` returns in under 0.5 s for a demo user; the results table exists; tests pass.
Baseline to beat (Acc@1, after the time-aware heuristic was merged): NYC 28.6, TKY 20.8, MS Tokyo 12.3. Earlier quick check with the plain counter: Markov 24.2 and LLM + counter + Markov 25.6 on NYC, so re-measure everything against the new heuristic.

### Member 3 (Aayush): LLM layer, all UI changes, integration

| # | Task | Output | Time |
|---|---|---|---|
| 3.1 | **Contract and stubs** (step 0): `contract.py`, plus a stub `gather()` returning realistic fake `Evidence`, so the UI can be built before M1/M2 finish | `tracex_assist/contract.py`, `tracex_assist/stub.py` | 20 min |
| 3.2 | **Ollama client**: status check (server running, model present), streaming chat via `http://127.0.0.1:11434/api/chat`, timeout, model choice (`qwen2.5:3b` default, `qwen2.5:7b` optional) | `tracex_assist/llm.py` | 30 min |
| 3.3 | **Grounded prompt**: system rules (use only the evidence, quote counts exactly, say when evidence is thin, offer "something new" only from the explore list) + the evidence JSON + the question | `llm.py` | 30 min |
| 3.4 | **Answer validator**: every place ID / name and every percentage in the answer must appear in the evidence; one retry with a stricter prompt, else the template answer. **Template fallback** used when Ollama is off | `llm.py` | 40 min |
| 3.5 | **Dashboard page "Personal assistant"**: dataset + persona picker (demo users first), profile card (top places, routine heatmap weekday × time, category mix), context pickers (day, time, current place from the user's places, "suggest something new"), chat with streaming answers, Ollama status badge | `app/views/assistant.py`, `app/dashboard.py` (nav entry) | 75 min |
| 3.6 | **"How I got this" panel** (facts with counts, back-off level, thin-evidence warning, offline accuracy) and **map** (current place, usual places, new suggestions) | `assistant.py`, `app/ui.py`, `app/tracex.css` | 40 min |
| 3.7 | **Results in the UI**: add Markov / recency / engine rows to the E1 page and a section for M1's copy analysis on Runs & behaviour (reads M1's and M2's report json) | `app/views/e1.py`, `app/views/runs.py`, `app/data.py` | 30 min |
| 3.8 | Tests (mocked Ollama: validator catches invented places and numbers, fallback used when offline, prompt contains only evidence), docs (`dashboard_guide.md` page 7, `llm_pipeline.md`, README) | `tests/test_assist_llm.py`, docs | 30 min |

**Done when:** the page works with Ollama on and off; the validator rejects a made-up answer; all tests pass.

---

## 5. Timeline

| Time | Member 1 | Member 2 | Member 3 (Aayush) |
|---|---|---|---|
| 0:00-0:20 | Read the contract | Read the contract | 3.1 contract + stubs, push |
| 0:20-2:30 | 1.1, 1.2, 1.3 | 2.1, 2.2, 2.3, 2.4 (use a tiny hand-made Profile until M1 pushes) | 3.2, 3.3, 3.4, start 3.5 on the stub |
| **2:30** | **Checkpoint 1:** push profile.py | **push evidence.py** | swap stub → real `gather`, run end to end |
| 2:30-3:45 | 1.4 copy analysis, 1.5 tests | 2.5 offline evaluation, 2.6 tests | 3.5, 3.6 |
| **3:45** | **Checkpoint 2:** push reports | **push reports** | 3.7 results in UI |
| 3:45-4:30 | Help test the page; pick demo scenarios | Help test the page; check numbers | 3.8 tests and docs |
| 4:30-5:00 | **All:** merge, full test run, 5-minute demo rehearsal (3 scenarios: a habit, "something new", thin evidence) | | |

If behind at checkpoint 1, drop in this order: map → routine heatmap → LLM fusion analysis → Tokyo support. The core
(engine + grounded answer + evidence panel on NYC) must not be dropped.

---

## 6. Git workflow

1. Everyone starts from the branch with the latest merged code (`main` if `Aayush` is merged, otherwise `Aayush`).
2. Branches: `assist-data` (M1), `assist-ir` (M2), `Aayush` (M3).
3. Member 3 pushes `contract.py` + stubs first; M1 and M2 pull it before starting.
4. Commit small and often, one task per commit (e.g. `Add place index with nearby search (assistant 1.1)`).
5. Merge order at the end: `assist-data` → `assist-ir` → `Aayush` → `main`, running `python -m pytest tests -q` after each.

---

## 7. Paper additions (each writes their own part)

| Member | Section |
|---|---|
| M1 | "What the LLM copies": the copy-analysis table and where the LLM beats the counter |
| M2 | Stronger baselines and the evidence engine: Markov / recency / fusion, category-level accuracy, the exploratory LLM fusion |
| M3 | The grounded assistant: design (IR decides, LLM explains), validator, fallback, example dialogues, limits |

**Limits to state:** NYC has no venue names; Tokyo has thin histories; suggestions of new places are not
evaluated for accuracy (they are popularity-based options, not predictions); the LLM fusion result is post hoc.

---

## 8. Report formats the dashboard reads (as produced; updated after integration)

All rates are fractions (0-1), like `reports/llm/evaluation.json`. A metric with a 95% bootstrap CI is
`{"mean": 0.286, "lo": 0.246, "hi": 0.326, "n": 500}`; a paired difference is
`{"diff": -0.024, "lo": -0.066, "hi": 0.018, "p": 0.28, "n": 500}`. Each dashboard section appears as soon as its file
exists and shows a short note until then. Keep these keys stable; adding keys is fine.

**Member 1: `reports/assistant/copy_analysis.json`** (`python -m tracex_assist.analysis_copy`), read by the
"What the LLM copies" table on Runs & behaviour (`app/views/runs.py`)

```json
{"sample": "sample_test_500", "n_boot": 10000, "alpha": 0.05,
 "rules": {"heuristic": "Time-aware heuristic", "counter": "Visit counter", "last": "Last place",
           "markov": "Markov", "trip": "Current trip", "any_rule": "Any of these"},
 "datasets": {
  "nyc": {"n": 500,
          "truth": {"heuristic": {mean, lo, hi, n}, "counter": {...}, "last": {...}, "markov": {...},
                    "trip": {...}, "any_rule": {...}},
          "random_place": {"heuristic": 0.06, ...}, "markov_coverage": 0.716, "counters": {...}}},
 "runs": {
  "nyc__qwen7b__llmmob__none__sample_test_500": {
    "dataset": "nyc", "model": "qwen7b", "method": "llmmob", "condition": "none", "main": true, "n": 500,
    "complete": true, "acc1": 0.232, "new_hits": {"hits": 0, "n": 109},
    "match":   {"heuristic": {mean, lo, hi, n}, "counter": ..., "last": ..., "markov": ..., "trip": ..., "any_rule": ...},
    "excess":  {"<rule>": {diff, lo, hi, p, n}},
    "outside": {"other_own": 0.178, "never_seen": 0.0, "invalid": 0.0},
    "ties": {...}, "disagree": {...}, "vs": {"<comparator>": {"<slice>": {...}}}}}}
```

The page uses, per run with `"main": true` (Massive-STEPS: names present), `match.<rule>.mean` (95% CI on hover) and
`outside.never_seen`, plus `datasets.<ds>.truth.<rule>` as the *Actual next place* reference row.

**Member 2: `reports/assistant/results.json`** (`python -m tracex_assist.evaluate`), read by the "Stronger
baselines and the assistant engine" section on E1 (`app/views/e1.py`) and by `evidence._offline_confidence`

```json
{"sample": "sample_test_500", "n_boot": 10000,
 "methods": {"frequency": {"nyc": {"dataset": "nyc", "n": 500, "acc1": {mean, lo, hi, n}, "acc5": ..., "mrr": ...,
                                   "ndcg5": ..., "cat_acc1": ..., "cat_acc5": ..., "revisit": {...}, "new": {...}},
                           "tky": {...}, "msteps_tokyo": {...}},
             "markov": {...}, "recency": {...}, "engine": {...}, "llm_fusion": {...}},
 "engine": {"nyc": {"acc1": 0.292, "acc5": 0.624, "n": 500}},
 "comparisons": [{"dataset": "nyc", "method": "markov", "n": 500, "acc1": {diff, lo, hi, p, n}, "acc5": ...,
                  "mcnemar_acc1": {"a_only": 51, "b_only": 63, "p": 0.30, "test": "chi2-cc", "p_holm": 1.0}}]}
```

Rows appear in the order of `methods`; `llm_fusion` is tagged *post hoc*. The page always adds the time-aware
heuristic (the E1 baseline, from `tracex_llm.evaluate`) as the first row. Optional `"comparator": "heuristic"` at the
top level names what `comparisons` are measured against (default: `frequency`, shown as "visit counter").

**Resolved at integration:** Member 2's evaluation now compares against the time-aware heuristic (method key
`heuristic`, from `tracex_ir.baselines.frequency_baseline`, identical to the E1 baseline) and builds profiles through
`tracex_assist.api` (Member 1's real module). With a `heuristic` row present the page labels the comparison
"vs time-aware heuristic" without needing a `comparator` key.

**Demo personas** (`profile.demo_users()[i]["persona"]`, Member 1): `from_poi → to_poi` happened `count` of `total`
times **across all days**; `weekday`/`hour` are one example moment when it happened (used by *Load this routine*),
not the usual time.
