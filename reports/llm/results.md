# TraceX LLM results (sample_test_500)

Values in %, 95% bootstrap CI in brackets (10000 resamples). Differences are paired over the same instances; p = McNemar on Acc@1, Holm-adjusted within each table. Copy = top-1 is a POI from the user's history or current trajectory. Invalid = no usable top-1 POI.

## E1: heuristic floor

| Dataset | Method | Model | n | Acc@1 | Acc@5 | MRR | NDCG@5 | Copy | Invalid | Δ Acc@1 vs freq | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msteps_tokyo | frequency | - | 334 | 16.8 [12.9, 21.0] | 38.9 [33.8, 44.3] | 0.265 [0.226, 0.306] | 0.286 [0.245, 0.329] | 100.0 | 0.0 | - | - |
| msteps_tokyo | llmmob | qwen7b | 334 | 19.2 [15.0, 23.7] | 37.1 [32.0, 42.2] | 0.267 [0.227, 0.310] | 0.288 [0.245, 0.331] | 100.0 | 0.0 | +2.4 [-1.2, +6.0] | 0.973 |
| msteps_tokyo | retrieval-jaccard | qwen7b | 334 | 18.9 [14.7, 23.1] | 32.9 [28.1, 38.0] | 0.247 [0.206, 0.289] | 0.263 [0.221, 0.306] | 99.1 | 0.0 | +2.1 [-1.8, +6.3] | 1.000 |
| nyc | frequency | - | 500 | 21.8 [18.2, 25.4] | 57.8 [53.4, 62.0] | 0.374 [0.341, 0.406] | 0.412 [0.377, 0.447] | 100.0 | 0.0 | - | - |
| nyc | llmmob | qwen7b | 500 | 23.2 [19.6, 27.0] | 55.0 [50.6, 59.2] | 0.363 [0.329, 0.397] | 0.401 [0.366, 0.437] | 100.0 | 0.0 | +1.4 [-2.2, +5.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | 500 | 22.8 [19.2, 26.6] | 50.0 [45.8, 54.4] | 0.339 [0.305, 0.374] | 0.370 [0.335, 0.406] | 99.4 | 0.0 | +1.0 [-3.2, +5.0] | 1.000 |

## E2: revisit vs new targets (Acc@1)

Revisit = the target POI appears in the user's history or current trajectory (fixed definition, docs/data_pipeline.md).

| Dataset | Method | Model | Subset | n | LLM Acc@1 | Freq Acc@1 | Δ vs freq | p |
|---|---|---|---|---|---|---|---|---|
| msteps_tokyo | llmmob | qwen7b | revisit | 158 | 40.5 [32.9, 48.1] | 35.4 [27.8, 43.0] | +5.1 [-2.5, +12.7] | 1.000 |
| msteps_tokyo | llmmob | qwen7b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen7b | revisit | 158 | 39.9 [32.3, 47.5] | 35.4 [27.8, 43.0] | +4.4 [-3.8, +13.3] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen7b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | llmmob | qwen7b | revisit | 391 | 29.7 [25.3, 34.3] | 27.9 [23.5, 32.2] | +1.8 [-2.8, +6.4] | 1.000 |
| nyc | llmmob | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | revisit | 391 | 29.2 [24.8, 33.8] | 27.9 [23.5, 32.2] | +1.3 [-4.1, +6.6] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |

## E3: venue-name test (Massive-STEPS)

| Model | Method | Condition | n | Acc@1 | Acc@5 | MRR | Copy | Invalid |
|---|---|---|---|---|---|---|---|---|
| qwen7b | llmmob | aliased | 334 | 20.4 [16.2, 24.9] | 37.1 [32.0, 42.5] | 0.276 [0.233, 0.319] | 100.0 | 0.0 |
| qwen7b | llmmob | present | 334 | 19.2 [15.0, 23.7] | 37.1 [32.0, 42.2] | 0.267 [0.227, 0.310] | 100.0 | 0.0 |
| qwen7b | llmmob | removed | 334 | 20.1 [15.9, 24.6] | 37.4 [32.3, 42.8] | 0.276 [0.234, 0.319] | 100.0 | 0.0 |
| qwen7b | retrieval-jaccard | aliased | 334 | 20.4 [16.2, 24.6] | 34.4 [29.3, 39.5] | 0.263 [0.221, 0.306] | 99.7 | 0.0 |
| qwen7b | retrieval-jaccard | present | 334 | 18.9 [14.7, 23.1] | 32.9 [28.1, 38.0] | 0.247 [0.206, 0.289] | 99.1 | 0.0 |
| qwen7b | retrieval-jaccard | removed | 334 | 19.8 [15.6, 24.0] | 34.4 [29.3, 39.5] | 0.256 [0.214, 0.299] | 99.1 | 0.0 |

| Model | Method | Comparison | n | Δ Acc@1 | Δ Acc@5 | p (Acc@1) |
|---|---|---|---|---|---|---|
| qwen7b | llmmob | present − aliased | 334 | -1.2 [-3.6, +1.2] | +0.0 [-2.1, +2.1] | 1.000 |
| qwen7b | llmmob | present − removed | 334 | -0.9 [-3.6, +1.8] | -0.3 [-2.4, +1.8] | 1.000 |
| qwen7b | llmmob | aliased − removed | 334 | +0.3 [-1.5, +2.1] | -0.3 [-2.4, +1.8] | 1.000 |
| qwen7b | retrieval-jaccard | present − aliased | 334 | -1.5 [-3.9, +0.9] | -1.5 [-3.9, +0.9] | 1.000 |
| qwen7b | retrieval-jaccard | present − removed | 334 | -0.9 [-3.3, +1.5] | -1.5 [-4.2, +1.2] | 1.000 |
| qwen7b | retrieval-jaccard | aliased − removed | 334 | +0.6 [-1.8, +3.0] | +0.0 [-2.7, +2.7] | 1.000 |
