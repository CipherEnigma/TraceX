# TraceX: what we set out to do, how we built it, and what we found

*Copy, Recall, or Reason?* A plain-language walkthrough of the whole project for the group, from the question to the
results. The technical details are in [data_pipeline.md](data_pipeline.md) (data),
[../tracex_ir](../tracex_ir) (retrieval), [llm_pipeline.md](llm_pipeline.md) (LLM and statistics) and
[dashboard_guide.md](dashboard_guide.md) (the dashboard, page by page).

---

## 1. The goal

**The task.** Next-POI (point of interest) prediction: given where a person has checked in so far today and their past
check-ins, predict the place they will check in at next. Location apps, maps and mobility planning all depend on this.

**Why it matters now.** Recent papers (LLM-Mob, LLM4POI, Nishida et al.) report that large language models do well at
this task just from a text prompt. But a good score alone does not say *why* the model is right. There are three
very different explanations:

| Hypothesis | What the model would be doing | What we would see |
|---|---|---|
| **Copy** | Repeating the places the user already visits most | It matches a simple visit counter and fails on places the user never visited |
| **Recall** | Using memorised knowledge of real venues from its training data | Accuracy drops when real venue names are hidden |
| **Reason** | Inferring the next place from time, order and category | It beats the counter and gets some never-visited places right |

**Our question:** which of these explains LLM next-POI accuracy? We answer it with three experiments:

- **E1 Heuristic floor:** does the LLM beat a frequency counter? (copy vs reason)
- **E2 Revisit vs new:** can it predict places the user has never been? (copy vs reason)
- **E3 Name test:** does it need real venue names? (recall)

This is an **information retrieval** project at heart: next-POI prediction is treated as *ranking* places for a
query (the current trip), retrieval is used to pick the examples the LLM sees, and everything is judged with IR
ranking metrics and significance tests on a fixed test collection.

---

## 2. How the project was built, step by step

The work was split into three parts that plug into each other:

```
Member 1: data pipeline  ->  Member 2: retrieval + baseline  ->  Member 3: LLM, experiments, dashboard
(clean data, splits)          (similar past trips, counter)        (prompts, runs, statistics, UI)
```

### Step 1. Data (Member 1, `tracex_data/`)

1. **Three datasets.** Foursquare NYC and Foursquare Tokyo (TKY), the standard benchmarks, and Massive-STEPS Tokyo,
   which also has **real venue names** (needed for E3).
2. **Cleaning.** Remove duplicates, missing and invalid records; keep users and places with enough check-ins.
3. **Trips (trajectories).** Each user's check-ins are sorted by time and cut into trips wherever there is a gap of
   24 hours or more.
4. **Time-based split.** 80% / 10% / 10% train / validation / test **by time**, so the model is always tested on the
   future, never the past.
5. **The prediction task.** For each test trip, hide its last check-in (the **target**) and predict it from the
   earlier check-ins of that trip (the **prefix**) plus the user's earlier trips (the **history**).
6. **No leakage.** History only contains trips that ended before the test trip started. A target is a
   **revisit** if the place appears in the history or the prefix, otherwise it is **new**.
7. **Evaluation samples.** A fixed random sample (seed 42) of 500 test cases for NYC and TKY, and all 334 for
   Massive-STEPS Tokyo, plus a 20-case debug sample. Every method is scored on exactly these cases.

### Step 2. Retrieval and baseline (Member 2, `tracex_ir/`)

1. **Time-aware frequency baseline (the "heuristic floor").** The brief's heuristic, with no AI at all: every past
   visit to a place counts, more if it happened around the same hour as the check-in being predicted, and less the
   longer ago it was (90-day half-life). The weights were tuned on the validation split (hour weight 4; the weekday
   weight came out as 0, it did not help). If an LLM cannot beat this, it is not adding much.
2. **Retrieval of similar past trips.** The current trip is the *query*, the user's past trips are the *documents*,
   and place IDs are the *terms*. Implemented: **Jaccard** (overlap of places), **LCS** (same places in the same order),
   **TF-IDF + cosine**, **BM25**, and a **hybrid** that merges rankings (reciprocal rank fusion).
3. **Metrics.** Precision@K, Recall@K, MAP, MRR, Accuracy@K and NDCG@K, tuned on the validation split only.
4. **The hand-over to Member 3.** `retrieve_demonstrations()` returns the most similar past trips together with where
   the user went next, ready to be shown to the LLM as examples.

### Step 3. The LLM pipeline (Member 3, `tracex_llm/`)

Built in small parts, each its own commit:

1. **Settings in one place** (`config.py`). Models: Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct in 4-bit, greedy
   decoding (no randomness), 10 ranked predictions per answer. Every run is named and stored by its settings.
2. **Two prompting methods** (`prompts.py`):
   - **LLM-Mob style:** show the user's 40 most recent past check-ins and the current trip, then ask for the next
     place.
   - **Retrieval demonstrations:** show the 5 most similar past trips (Jaccard, from Member 2), each ending with
     "→ next: …", then the current trip, then ask.
   - Both ask for a JSON answer: `{"prediction": [10 POI IDs], "reason": "one sentence"}`.
   - **Leakage rule:** the prompt may contain the *time* of the next check-in (as in the published methods) but never
     its place, category or name. Checked on all 2,668 prompts and by automated tests.
3. **Reading the answers** (`parse.py`). JSON first, then fallbacks for messy answers. Places that do not exist stay in
   their position and count as misses, and short answers are never padded, so no method gets a free boost. Each answer
   is also flagged: is it invalid? is the first guess a place the user already visited (**copy**)?
4. **Running the model** (`backends.py`, `run.py`). Batched 4-bit generation on a GPU. Runs are **resumable**: every
   batch is saved immediately, a crashed run continues where it stopped, and a run refuses to continue with different
   settings, so a file never mixes two configurations.
5. **E3 name disguise** (`aliasing.py`). Three versions of the Massive-STEPS test set: real names, no names, and fake
   names. Each venue gets one consistent fake katakana name of similar length (so the script still looks Japanese and
   only the meaning is removed). We checked that the three prompts differ *only* in the names.
6. **Statistics** (`stats.py`, `evaluate.py`). Every method is compared on the same test cases:
   - 95% bootstrap confidence intervals (10,000 resamples),
   - paired bootstrap for differences,
   - McNemar's test (did method A get right the cases B got wrong more often than chance?),
   - Holm correction, because we run many comparisons.
7. **Figures** (`report.py`) for the paper, in `reports/llm/figures/`.

### Step 4. Running the experiments on a GPU (Kaggle)

Qwen-7B needs a GPU, so all runs used **one free Kaggle Tesla T4** for both models (same hardware = fair speed and
accuracy comparison). The notebook `notebooks/kaggle_run.ipynb` downloads the code, rebuilds the data, checks the
samples are identical to the repo's, and runs everything.

1. **Debug run** (20 cases) to check the answers parse and to measure speed.
2. **Gate 1, the reproduction check.** Qwen-7B on NYC must land near the published 18-23% Accuracy@1, otherwise
   something is broken. Result: **23.2%, passed.**
3. **E3 with Qwen-7B**, then one **all-in-one run** (`STAGE = "all"`) that did the rest: 7B on TKY, then Qwen-3B on
   everything, with a sanity check before the long runs, and the evaluation at the end.
4. In total: **20 runs and 8,008 predictions** (4,004 per model, all completed), about 3.4 s per prediction for 7B
   and 1.9 s for 3B, roughly 6 GPU-hours.

### Step 5. Dashboard, tests and documentation

- **Streamlit dashboard** (`app/`): six pages, designed from Claude Design boards, reading the results live. See
  [dashboard_guide.md](dashboard_guide.md).
- **Tests** (`tests/test_llm.py`): 27 tests that run without a GPU (leakage, parsing, aliases, statistics, resuming).
  With Members 1 and 2's tests: 40 passing.
- **Documentation:** this file, `llm_pipeline.md` (every setting and why), `dashboard_guide.md`.

### Step 6. A personal assistant built on the findings

Because the LLM copies rather than reasons, the last piece uses each tool for what it is good at: an **evidence
engine** (Member 1's user profiles, Member 2's back-off statistics and rank fusion, both pure IR) decides where a user
is likely to go and why, and **Qwen2.5-3B, run locally with Ollama, only explains** that evidence. A validator
checks that every place and number in the answer comes from the evidence, with a template answer as a safety net.
It is page 7 of the dashboard; the plan and division of work are in [assistant_plan.md](assistant_plan.md).

---

## 3. What we found

Accuracy@1 (%), 500 test cases (334 for Massive-STEPS Tokyo):

| | NYC | TKY | Massive-STEPS Tokyo |
|---|---|---|---|
| Time-aware frequency heuristic | **28.6** | **20.8** | 12.3 |
| Qwen-7B LLM-Mob / Retrieval | 23.2 / 22.8 | 16.0 / 17.0 | **19.2** / 18.9 |
| Qwen-3B LLM-Mob / Retrieval | 16.6 / 17.4 | 13.6 / 14.6 | 12.3 / 15.3 |

1. **E1: a tuned habit counter beats the LLM when there is enough history.** On NYC and TKY the time-aware heuristic
   beats Qwen-7B by 4-6 points (significant on NYC for both methods, p ≈ 0.04, and on TKY LLM-Mob, p = 0.021) and
   Qwen-3B by 6-12 points (p ≤ 0.015). On Massive-STEPS Tokyo, where users have very short histories (median 10
   check-ins), Qwen-7B beats it by about 7 points (p ≤ 0.021): with little to count, the counter is weak, and the LLM's
   ranking of the few known places helps.
2. **E2: copying, not reasoning.** Both models got **0 of 1,272** never-visited places right, and 90-100% of first
   guesses are places the user already visited. All of the accuracy comes from revisits: on revisit targets the counter
   wins on NYC and TKY (by about 7 points) and 7B wins on Tokyo (by about 14 points).
3. **E3: no recall of venue names.** Real, removed and fake names give the same accuracy (all differences within ±1.5
   points, p = 1.00). The models are not using knowledge of the actual venues.
4. **Model size.** 7B beats 3B by 2-8 points (significant on NYC and Massive-STEPS LLM-Mob). The bigger model is better
   at choosing *which* of the user's places comes next, but neither predicts anything new.

5. **The assistant engine (pure IR, no LLM)** matches the strong heuristic where users have a rich history (NYC
   29.2 vs 28.6, TKY 19.8 vs 20.8, both not significant) and clearly beats it where history is thin (Massive-STEPS
   Tokyo 22.2 vs 12.3, p < 0.001). Fusing the saved LLM rankings into it does not help (exploratory). So the LLM is
   used only to explain the engine's evidence in the assistant.

**Answer to "Copy, Recall, or Reason?": Copy.** On these benchmarks, Qwen2.5 LLMs predict the next place by
re-ranking the user's own habits. With a rich history, a simple time-aware counter does that job better; only when the
history is thin does the LLM's re-ranking add something. They never predict new places and do not depend on venue
names.

---

## 4. Limits (to state honestly in the paper)

- The prompt includes the time of the next check-in, like the published methods; the place itself is never shown.
- Retrieval prompts sometimes show fewer than 10 distinct places, so the model lists fewer, which limits Acc@5.
- 4-bit quantisation and batched decoding can differ slightly from full-precision results.
- Massive-STEPS Tokyo has only 334 test cases, so its intervals are wider.
- Two Qwen sizes only; larger or differently trained models may behave differently.
- The name test shows the models do not *rely* on names; it does not prove what is or is not in their training data.

---

## 5. Glossary

| Term | Meaning |
|---|---|
| POI | Point of interest: a venue such as a café, station or gym, identified by a POI ID |
| Check-in | A record that a user was at a POI at a given time |
| Trajectory / trip | A user's check-ins with no gap of 24 h or more |
| Prefix | The check-ins of the current trip before the one we predict |
| Target | The check-in we try to predict (the last one of a test trip) |
| History | The user's earlier trips that ended before the test trip started |
| Revisit / new | The target place is / is not in the history or prefix |
| Frequency heuristic | Baseline that ranks the user's visited places, weighted by time of day and recency |
| Demonstration | A similar past trip shown to the LLM as a worked example |
| Acc@K | The true place is among the top K guesses |
| MRR | Average of 1 / rank of the true place |
| NDCG@5 | Top-5 score that counts higher-ranked hits more |
| Copy rate | Share of first guesses that are places the user already visited |
| 95% CI | Range that would contain the true value in 95% of repeats of the experiment |
| McNemar / Holm | Paired significance test / correction for running many tests |
| Gate 1 | Our check that 7B reproduces published NYC accuracy before trusting the rest |
