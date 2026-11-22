# What the LLM copies

## Method

Experiments E1 and E2 show that the models' first guesses are almost always places the user has already visited. To
find out *which* of these places they choose, we compared the top-1 answer of every saved run with five rules computed
from the same input:

- the **time-aware heuristic** (visits weighted by hour and recency),
- a plain **visit counter**,
- the **last place** visited,
- the user's **Markov** top-1 (the place they most often went to after the last place, with no fallback),
- membership of the **current trip**.

The rules overlap, so their shares do not add up to 100%. As references we report how often the true next place follows
each rule, and how often a place drawn uniformly from the user's own places would. Differences in accuracy are paired
over instances, with 95% bootstrap intervals and McNemar tests, Holm-corrected within each comparator and grouping.

## Results

**Table: share of top-1 answers equal to each rule (%).**

| Dataset | Answer | Heuristic | Counter | Last place | Markov | Current trip | Any rule |
|---|---|---:|---:|---:|---:|---:|---:|
| NYC | True next place | 28.6 | 21.8 | 9.4 | 18.4 | 27.2 | 49.4 |
| | Random own place | 6.0 | 6.0 | 6.0 | 4.1 | 13.1 | 19.4 |
| | Qwen-7B, LLM-Mob | 44.0 | 46.8 | 26.6 | 20.6 | 50.4 | 82.2 |
| | Qwen-7B, retrieval | 38.0 | 35.0 | 34.2 | 21.0 | 63.8 | 83.0 |
| | Qwen-3B, LLM-Mob | 38.0 | 35.8 | 39.8 | 15.6 | 62.2 | 80.4 |
| | Qwen-3B, retrieval | 32.8 | 30.4 | 42.2 | 18.6 | 62.4 | 78.2 |
| TKY | True next place | 20.8 | 19.2 | 8.4 | 18.2 | 24.6 | 41.2 |
| | Random own place | 3.2 | 3.2 | 3.2 | 2.1 | 8.3 | 11.7 |
| | Qwen-7B, LLM-Mob | 46.6 | 48.2 | 24.2 | 18.8 | 50.2 | 78.8 |
| | Qwen-7B, retrieval | 35.0 | 36.0 | 29.2 | 20.6 | 66.4 | 84.0 |
| | Qwen-3B, LLM-Mob | 36.2 | 33.4 | 38.8 | 14.0 | 66.2 | 81.4 |
| | Qwen-3B, retrieval | 26.6 | 22.4 | 38.0 | 16.8 | 63.0 | 75.8 |
| Massive-STEPS Tokyo | True next place | 12.3 | 16.8 | 4.5 | 12.6 | 9.9 | 29.9 |
| | Random own place | 16.2 | 16.2 | 16.2 | 5.9 | 21.4 | 33.9 |
| | Qwen-7B, LLM-Mob | 38.0 | 55.4 | 24.6 | 11.7 | 33.5 | 71.3 |
| | Qwen-7B, retrieval | 28.7 | 36.2 | 18.9 | 18.9 | 30.8 | 63.8 |
| | Qwen-3B, LLM-Mob | 50.3 | 40.4 | 47.9 | 8.7 | 64.1 | 78.7 |
| | Qwen-3B, retrieval | 27.8 | 24.3 | 23.7 | 15.9 | 30.5 | 54.8 |

**The models over-copy simple signals.** On NYC and TKY, 76-84% of first guesses coincide with at least one rule,
whereas only 49% and 41% of true next places do. On Massive-STEPS the figures are 55-79% against 30%. Among the
individual rules, the largest excess over the truth is for the current trip in nine of the twelve E1 runs, and for the
visit counter in the three Qwen-7B LLM-Mob runs. The models also answer with the last place in 19-48% of cases, although
it is the true next place in only 4-9%. In contrast, agreement with the user's transition habits (Markov) stays within
about 6 points of the rate in the true data. The models over-copy what is frequent or recent much more than what
usually comes next.

**The two model sizes copy differently.** Where the counter and the last place disagree, Qwen-7B with LLM-Mob follows
the counter in 44-56% of cases and the last place in 10-17%. Qwen-3B with LLM-Mob leans the other way: 26-28% counter
against 33-39% last place. Retrieved demonstrations lower the agreement with the counter for both models.

**Where the LLM beats the counter.** Against the plain visit counter, no run is significantly better, either overall or
in any subgroup by history length, time of day, or revisit versus new. Against the time-aware heuristic, the only
significant wins are Qwen-7B's on Massive-STEPS: +6.9 points with LLM-Mob (+14.6 on revisits) and +6.6 with retrieval.
On never-visited targets the E1 runs got 0 of 1,696 first guesses right.

**The Massive-STEPS win depends on the baseline.** With very short histories, the time-aware heuristic is dominated by
the current trip: its first guess is the place the user is already at in 66% of cases, and it scores 12.3%, below the
plain counter's 16.8%. In an exploratory check we computed the same heuristic from the history alone. It scores higher
on validation on all three datasets (28.1 vs 27.6 on NYC, 20.9 vs 20.2 on TKY, 24.8 vs 15.8 on Massive-STEPS) and
reaches 29.6, 21.6 and 23.4% on test. All twelve E1 runs score below this variant, nine of them significantly, and
Qwen-7B's advantage on Massive-STEPS disappears (-4.2 and -4.5 points, neither significant).

## Implication for the assistant

We found no evidence that the LLM selects the next place better than well-built counting does. This supports the
assistant's division of work: statistics over the user's history select and justify the places, and the LLM words that
evidence.

## Limits

- Agreement with a rule describes the output. It does not show that the model computes that rule.
- Ties go to the lowest POI id, which understates agreement with the counter when histories are short. Counting any
  tied place raises it from 55.4% to 76.9% for Qwen-7B with LLM-Mob on Massive-STEPS.
- Markov is defined for 72% of NYC, 66% of TKY and 35% of Massive-STEPS instances, and an undefined case counts as no
  match.
- The history-only comparator was proposed after inspecting test behaviour. The validation split supports it, but it
  was not pre-registered.
- Many subgroups are small (as few as 5 instances), so the absence of a significant win there is weak evidence of no
  difference.
- All methods are given the time of the check-in they predict.
