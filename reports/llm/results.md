# TraceX LLM results (sample_test_500)

Values in %, 95% bootstrap CI in brackets (10000 resamples). Differences are paired over the same instances; p = McNemar on Acc@1, Holm-adjusted within each table. Copy = top-1 is a POI from the user's history or current trajectory. Invalid = no usable top-1 POI.

## E1: heuristic floor

| Dataset | Method | Model | n | Acc@1 | Acc@5 | MRR | NDCG@5 | Copy | Invalid | Δ Acc@1 vs freq | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nyc | frequency | - | 500 | 21.8 [18.2, 25.4] | 57.8 [53.4, 62.0] | 0.374 [0.341, 0.406] | 0.412 [0.377, 0.447] | 100.0 | 0.0 | - | - |
| nyc | llmmob | qwen7b | 500 | 23.2 [19.6, 27.0] | 55.0 [50.6, 59.2] | 0.363 [0.329, 0.397] | 0.401 [0.366, 0.437] | 100.0 | 0.0 | +1.4 [-2.2, +5.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | 500 | 22.8 [19.2, 26.6] | 50.0 [45.8, 54.4] | 0.339 [0.305, 0.374] | 0.370 [0.335, 0.406] | 99.4 | 0.0 | +1.0 [-3.2, +5.0] | 1.000 |

## E2: revisit vs new targets (Acc@1)

Revisit = the target POI appears in the user's history or current trajectory (fixed definition, docs/data_pipeline.md).

| Dataset | Method | Model | Subset | n | LLM Acc@1 | Freq Acc@1 | Δ vs freq | p |
|---|---|---|---|---|---|---|---|---|
| nyc | llmmob | qwen7b | revisit | 391 | 29.7 [25.3, 34.3] | 27.9 [23.5, 32.2] | +1.8 [-2.8, +6.4] | 1.000 |
| nyc | llmmob | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | revisit | 391 | 29.2 [24.8, 33.8] | 27.9 [23.5, 32.2] | +1.3 [-4.1, +6.6] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
