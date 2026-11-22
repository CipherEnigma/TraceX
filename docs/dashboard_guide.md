# TraceX dashboard: page-by-page guide

The dashboard shows every result of the project in one place: the three experiments (E1, E2, E3), single
predictions you can inspect one by one, the health of every LLM run, and a personal assistant built on what we
learned (page 7). It reads the saved results in
`results/llm/` and uses the same evaluation code as `reports/llm/results.md`, so every number on screen matches
the paper tables.

## Starting it

```powershell
cd TraceX
pip install -r requirements.txt streamlit     # first time only; no GPU packages needed
streamlit run app/dashboard.py                 # opens http://localhost:8501
```

Stop it with **Ctrl + C** in the terminal. New result files appear after a page refresh.

The Personal assistant page (7) also uses [Ollama](https://ollama.com) if it is installed and running, with the
model `qwen2.5:3b` (`ollama pull qwen2.5:3b`, about 1.9 GB). Without Ollama the page still works and shows
template answers built from the same evidence.

## Things that work the same on every page

| Element | What it does |
|---|---|
| **Sidebar: page list** | The six pages below. The blue bar marks the current page. |
| **Sidebar: Sample** | Which test sample the numbers come from. `sample_test_500` is the real evaluation (500 instances for NYC and TKY, 334 for Massive-STEPS Tokyo). `sample_test_debug20` is the 20-instance smoke test. |
| **Sidebar: Results last updated** | Time of the newest result file. |
| **⋮ menu (top right)** | Switch **System / Light / Dark** theme. |
| **» / « (top left)** | Collapse or expand the sidebar. |
| **Filters row** | Every page has its filters in one row under the title. Changing one redraws the page. |
| **Hovering a bar** | Shows a tooltip with the exact value, the 95% interval and the number of instances. |

**How to read the numbers everywhere**

- Values are **percentages**. MRR and NDCG are scores from 0 to 1; the dashboard shows them ×100 (36.3 = 0.363 in
  `reports/llm/results.md`).
- `23.2 [19.6, 27.0]` = the value, then its **95% confidence interval** in brackets. If two intervals overlap a lot,
  the difference may just be luck.
- **Δ (delta)** = a difference in percentage points between two methods on the *same* test instances.
- **p** = p-value from McNemar's test, already **Holm-corrected** for running many comparisons. Below 0.05 means the
  difference is unlikely to be chance.
- **Frequency heuristic** = the non-AI baseline: rank the places the user has visited, counting visits at a similar
  hour more and old visits less (the brief's time-aware heuristic). It is drawn as a
  dashed line with a grey band (its 95% interval).
- **7B / 3B** = Qwen2.5-7B-Instruct (blue) and Qwen2.5-3B-Instruct (orange), both 4-bit, same GPU.
- **LLM-Mob** = the model sees the user's recent history. **Retrieval** = the model sees the user's most similar past
  trips (found with Jaccard similarity) as worked examples.

---

## 1. Overview

**Question it answers:** what is the project asking, and what is the headline answer?

**What is on the page, top to bottom**

1. **Filters:** Dataset, Model, Method. They narrow the status table at the bottom.
2. **Three hypotheses: Copying, Recall, Reasoning.** What each would look like in the data and which
   experiment tests it (links jump to that page):
   - *Copying:* the model repeats places the user already visits. Tested by E1 and E2.
   - *Recall:* the model uses memorised knowledge of real venues. Tested by E3.
   - *Reasoning:* the model infers the next place from time, order and category. Tested by E1 and E2.
3. **Best LLM vs frequency heuristic (one card per dataset).** For each dataset the best LLM setting is compared
   with the heuristic on Accuracy@1:
   - the two values with their intervals,
   - the paired difference Δ with its interval, drawn on a small −10 to +10 scale (the dot is Δ, the line is its
     interval, the tick is zero),
   - a one-line verdict: *Above the floor*, *Below the floor* or *CI crosses zero: no better than counting visits*.
4. **Gate 1 banner.** Our reproduction check: does Qwen-7B on NYC land near the published 18-23% Accuracy@1? It shows
   *Passed* (23.2%). This tells us the pipeline is not broken before we trust the other experiments.
5. **Experiment status table.** Every planned run (dataset × method × model × name condition) and whether it is
   complete for E1, E2 and E3. "n/a" means that experiment does not use that run.

**What it shows right now:** all 20 runs complete. The best LLM is significantly
below the heuristic on NYC and TKY ("Below the floor") and above it on Massive-STEPS Tokyo, where users have very
short histories.

---

## 2. E1 Heuristic floor

**Question it answers:** does an LLM beat a simple counter of the user's most-visited places?

**Filters:** Dataset (NYC, TKY, Massive-STEPS Tokyo, All), Metric (Acc@1, Acc@5, MRR, NDCG@5), Model, Method.

**Chart.** One group of bars per prompting method, one bar per model, with 95% interval whiskers. The grey band is
the heuristic's interval. **How to read it:** if a bar's whisker overlaps the grey band, that LLM is not
distinguishable from simply counting visits.

**Results table.** Every metric for the heuristic and each LLM run, plus Copy %, Invalid %, Δ Acc@1 vs the heuristic
and its p-value.

**Metric meanings**

| Metric | Meaning |
|---|---|
| Acc@1 | The true next place is the model's first guess |
| Acc@5 | The true next place is in the top 5 guesses |
| MRR | Average of 1 / (rank of the true place); rewards ranking it high (cut at 10) |
| NDCG@5 | Like Acc@5 but a hit at rank 1 counts more than a hit at rank 5 |
| Copy % | Share of first guesses that are places the user already visited |
| Invalid % | Share of answers with no usable prediction (unreadable, empty, or a POI that does not exist) |

**What it shows right now:** the heuristic beats Qwen-7B on NYC (28.6 vs 23.2, p = 0.035) and TKY (20.8 vs 16.0-17.0),
and Qwen-7B beats it on Massive-STEPS Tokyo (19.2 vs 12.3, p = 0.015), where there is little history to count.
Qwen-3B is clearly below it on NYC and TKY and level with it on Tokyo.

**Stronger baselines and the assistant engine** (below each dataset's table, once
`reports/assistant/results.json` exists): non-LLM rankers on the same test cases: the time-aware heuristic,
**Markov** (where the user went next after their current place), **recency**, and the assistant's fused engine, with
Δ vs the heuristic and p-values. A second table shows how often the *kind* of place (category) is predicted
correctly. Rows tagged *post hoc* were designed after seeing the LLM results and are exploratory.

---

## 3. E2 Revisit vs new

**Question it answers:** can the LLM predict a place the user has **never** been, or only places it has seen?

**Filters:** Dataset, Model, Method.

**Tiles.** How the test targets split: **revisit** (the true next place is already in the user's history or current
trip) versus **new** (never visited before). For NYC: 78.2% revisit, 21.8% new.

**Chart.** Accuracy@1 on revisit targets and on new targets, for the heuristic and each LLM method. New-target bars are
labelled ("0.0, 0/109") because they sit at zero.

**Table.** LLM vs heuristic *within* each subset, with paired Δ and p-value.

**Definition of history (expandable box).** The exact rule for "revisit", shared with the Instance explorer and the
copy rate.

**What it shows right now:** both models get **0 correct out of 1,272 new-place targets** across all datasets. All
of their accuracy comes from revisits. The heuristic also scores 0 on new places by construction (it can only rank
visited places), so on new places the LLM adds nothing.

---

## 4. E3 Name test

**Question it answers:** does the model rely on knowing the real venues (memorised names), or only on the
history pattern?

**Dataset:** fixed to Massive-STEPS Tokyo, the only dataset with real venue names. Filters: Model, Method.

**The three conditions** (same 334 test cases, same history, same targets, same POI IDs and categories; only the
names change):

| Condition | What the model sees |
|---|---|
| present | Real venue names, e.g. "吉野家" |
| removed | No names at all |
| aliased | A consistent fake katakana name per venue, e.g. "サアデイクチイーレガ" |

**Chart.** Accuracy@1 per condition, with the heuristic band for reference.
**How to read it:** if accuracy drops when names are removed *and* fake names do not bring it back, the model was using
real-world knowledge of those venues.

**Paired comparisons table.** present − aliased, present − removed, aliased − removed, with Δ and p for Acc@1 and
Acc@5.

**See the difference.** Pick a test case and see the same trip shown three ways, side by side, with the venue-name parts
tinted. It proves only the names differ.

**What it shows right now:** no effect. Every difference is within ±1.5 points, p = 1.00, for both models and both
methods. The models do not use venue names here.

---

## 5. Instance explorer

**Question it answers:** what exactly did each model see and predict for one test case, and why?

**Filters:** Dataset, Models shown (7B + 3B, 7B, 3B), Names (Massive-STEPS only: present / removed / aliased).

**Left column: choose a case**

- **Search** by instance ID (e.g. `nyc-test-22`) or user ID.
- **Quick filter** with counts: All, LLM hit, LLM miss, Revisit, New, Methods disagree.
- **Previous / Next** to step through the filtered list.

**Main column**

1. **Instance header:** instance ID, user, the time of the check-in to predict, target type (revisit/new) and the
   size of the user's history.
2. **Reveal target** switch. Off by default so you can guess first. When on, the true place is highlighted in every
   list and the hit@1 / hit@5 badges are filled in.
3. **History** (newest first, grouped by trip) and **Current trajectory**, ending in the empty
   "Next: ?" slot that is being predicted.
4. **Predictions · top 10:** one card per method: frequency heuristic, LLM-Mob 7B/3B, Retrieval 7B/3B. Each place shows
   its ID and category, plus a **seen** tag if the user has been there before. A card may list fewer than 10 places
   because the model only lists places it can see in its prompt.
5. **LLM answer** (one tab per run): the model's own one-sentence reason, the parse status, output tokens and time,
   the **raw output**, and the **full prompt**. The prompt is rebuilt by the dashboard and checked against the
   fingerprint saved during the run ("✓ identical to the prompt used in the run").

**Good things to try in a demo:** filter *New* and look at what the models guess (always seen places), filter
*Methods disagree* to compare reasoning, and on Massive-STEPS switch *Names* to see the aliased prompt.

---

## 6. Runs & behaviour

**Question it answers:** did every run finish correctly, what did it cost, and how do the models behave?

**Filters:** Dataset, Model, Method.

1. **Compute tile:** average seconds per test case on the Tesla T4 (7B 3.36 s, 3B 1.93 s) and the projected GPU-hours.
2. **Instances completed:** 8,008 of 8,008 across 20 runs, with a progress bar.
3. **All runs table:** for every run: progress, seconds per instance, average answer length in tokens, GPU, a
   **settings fingerprint** (two runs with the same fingerprint used identical settings), and when it finished.
4. **Copy and invalid rates** with 95% intervals:
   - *Copy rate:* how often the first guess is a place the user already visited (90-100% everywhere).
   - *Invalid rate:* how often the answer was unusable (0-0.3%).
5. **Parse status:** how each answer was read: clean JSON (dark), regex fallback (grey) or failed (red). Almost
   everything is clean JSON; Qwen-3B had one empty answer and one regex fallback in 8,108 answers.
6. **What the LLM copies** (once `reports/assistant/copy_analysis.json` exists): for every LLM run, how often its
   first guess is exactly what a simple rule would pick: the heuristic's top place, the Markov top place, the last
   place visited, or a place from the current trip. High shares mean copying.

---

## 7. Personal assistant

**Question it answers:** if we use what we learned (IR decides, the LLM explains), what does a useful, honest
next-place assistant look like?

**Filters:** Dataset (NYC, TKY, Massive-STEPS Tokyo) and **User**: you role-play a real user from the data (the
list shows their check-in count and most common kind of place). A tag on the right shows whether the local LLM is
ready (**● Qwen2.5-3B ready**) or the page is using template answers.

**Left column: the moment**

1. **Day, Time, You are at** (one of the user's usual places) and **Suggest something new** (also list nearby
   places the user has never visited).
2. **Profile:** check-ins, trips, the period covered, and the 5 usual places.
3. **Weekly routine:** a heatmap of check-ins by weekday × time of day (morning, midday, afternoon, evening,
   night); darker = more. The cell for the chosen moment is outlined. Hover a cell for the count and the most common
   kind of place.

**Right column: ask**

1. Type a question or click one of the suggested ones. The question is read for intent: "something **new**",
   and the **kind** of place (food, coffee, drinks, shopping, outdoors, transport), so "somewhere new to eat"
   suggests restaurants the user has never been to.
2. **The answer** in plain words, with a line saying who wrote it (Qwen2.5-3B, Qwen2.5-3B after one stricter retry,
   or the template), how long it took, and that every place and number was checked against the evidence.
3. **How I got this** (expandable): the evidence table (what came next after this place, share of all check-ins,
   visits at this time of week), each with its count ("12 of 36"), share bar and the data it is based on; the nearby
   places the user has never visited with distance and popularity; a warning if the evidence rests on fewer than 5
   visits; the engine's offline accuracy; and **exactly what the model was shown**.
4. **Map:** you are here (orange), the user's usual places (blue) and suggested new places (green).

**How the answer is kept honest:** the evidence engine computes every place, count and distance from the user's
history; the model only rewrites it. A validator rejects any answer that mentions a place, percentage, count or
distance that is not in the evidence (or puts a distance next to the wrong place); the model then gets one stricter
retry, and after that the template answer is shown. In a test of 50 varied questions, 94% of answers passed on the
first try, 98% after the retry, and 2% fell back to the template.

**Good things to try in a demo:** a habit ("Where should I go next?"), something new ("Somewhere new to eat?"),
Massive-STEPS Tokyo for real venue names, and a quiet time of week to see the thin-evidence warning.

**Data limits:** NYC and TKY have no venue names (places are shown as "Coffee Shop #345"); Massive-STEPS Tokyo has
real names but short histories. Suggestions of new places are popularity-based options, not predictions.

---

## Where the same results live outside the dashboard

| What | File |
|---|---|
| All tables (E1, E2, E3, model size) | `reports/llm/results.md` and `reports/llm/evaluation.json` |
| Paper figures (PNG + PDF) | `reports/llm/figures/` |
| Raw predictions, one line per test case | `results/llm/<dataset>/*.jsonl` |
| How everything is computed | `docs/llm_pipeline.md` |
| The story of the project | `docs/project_overview.md` |
| Assistant engine evaluation and copy analysis | `reports/assistant/` |
| Assistant plan, contract and report formats | `docs/assistant_plan.md` |
