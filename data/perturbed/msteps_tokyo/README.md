# TraceX perturbed test sets: msteps_tokyo (sample_test_500)

The E3 name test of TraceX ("Copy, Recall, or Reason?"). 334 test instances in three versions that differ
**only in venue names**. History, targets, POI ids, categories and times are identical.

| File | Venue names |
|---|---|
| `sample_test_500_present.jsonl` | real Massive-STEPS names |
| `sample_test_500_removed.jsonl` | no `name` field |
| `sample_test_500_aliased.jsonl` | consistent fake aliases from `alias_map.csv` |

Each line is one instance: `prefix` (input check-ins), `target` (the check-in to predict), `history`
(the user's earlier training trajectories, oldest first) and the scoring flags `target_in_history`,
`target_in_prefix` and `is_revisit`. Never put `target` or the flags in a prompt.

Aliases: one per `poi_id` (seed 42, generator `katakana-v1`), random katakana pseudo-words similar in
length and script to the real names. Each is unique and never equals or occurs inside a real venue name.
Chain branches get different aliases, so chain identity is removed as well.

Regenerate with `python -m tracex_llm.aliasing export`.
