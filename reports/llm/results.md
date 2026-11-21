# TraceX LLM results (sample_test_500)

Values in %, 95% bootstrap CI in brackets (10000 resamples). Differences are paired over the same instances; p = McNemar on Acc@1, Holm-adjusted within each table. Copy = top-1 is a POI from the user's history or current trajectory. Invalid = no usable top-1 POI.

## E1: heuristic floor

| Dataset | Method | Model | n | Acc@1 | Acc@5 | MRR | NDCG@5 | Copy | Invalid | Δ Acc@1 vs freq | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| msteps_tokyo | frequency | - | 334 | 16.8 [12.9, 21.0] | 38.9 [33.8, 44.3] | 0.265 [0.226, 0.306] | 0.286 [0.245, 0.329] | 100.0 | 0.0 | - | - |
| msteps_tokyo | llmmob | qwen3b | 334 | 12.3 [9.0, 15.9] | 35.3 [30.2, 40.7] | 0.216 [0.181, 0.253] | 0.245 [0.207, 0.285] | 99.1 | 0.0 | -4.5 [-8.4, -0.9] | 0.295 |
| msteps_tokyo | retrieval-jaccard | qwen3b | 334 | 15.3 [11.4, 19.2] | 32.0 [26.9, 37.1] | 0.225 [0.187, 0.265] | 0.244 [0.203, 0.286] | 90.4 | 0.0 | -1.5 [-5.7, +3.0] | 1.000 |
| msteps_tokyo | llmmob | qwen7b | 334 | 19.2 [15.0, 23.7] | 37.1 [32.0, 42.2] | 0.267 [0.227, 0.310] | 0.288 [0.245, 0.331] | 100.0 | 0.0 | +2.4 [-1.2, +6.0] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen7b | 334 | 18.9 [14.7, 23.1] | 32.9 [28.1, 38.0] | 0.247 [0.206, 0.289] | 0.263 [0.221, 0.306] | 99.1 | 0.0 | +2.1 [-1.8, +6.3] | 1.000 |
| nyc | frequency | - | 500 | 21.8 [18.2, 25.4] | 57.8 [53.4, 62.0] | 0.374 [0.341, 0.406] | 0.412 [0.377, 0.447] | 100.0 | 0.0 | - | - |
| nyc | llmmob | qwen3b | 500 | 16.6 [13.4, 19.8] | 48.8 [44.4, 53.0] | 0.305 [0.275, 0.335] | 0.335 [0.302, 0.368] | 99.0 | 0.0 | -5.2 [-9.2, -1.4] | 0.137 |
| nyc | retrieval-jaccard | qwen3b | 500 | 17.4 [14.2, 20.8] | 48.6 [44.2, 53.0] | 0.302 [0.271, 0.334] | 0.339 [0.305, 0.373] | 99.4 | 0.0 | -4.4 [-8.4, -0.2] | 0.362 |
| nyc | llmmob | qwen7b | 500 | 23.2 [19.6, 27.0] | 55.0 [50.6, 59.2] | 0.363 [0.329, 0.397] | 0.401 [0.366, 0.437] | 100.0 | 0.0 | +1.4 [-2.2, +5.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | 500 | 22.8 [19.2, 26.6] | 50.0 [45.8, 54.4] | 0.339 [0.305, 0.374] | 0.370 [0.335, 0.406] | 99.4 | 0.0 | +1.0 [-3.2, +5.0] | 1.000 |
| tky | frequency | - | 500 | 19.2 [15.8, 22.6] | 43.0 [38.6, 47.4] | 0.292 [0.259, 0.325] | 0.314 [0.280, 0.349] | 100.0 | 0.0 | - | - |
| tky | llmmob | qwen3b | 500 | 13.6 [10.8, 16.6] | 36.0 [31.8, 40.2] | 0.229 [0.200, 0.259] | 0.252 [0.221, 0.285] | 98.6 | 0.0 | -5.6 [-8.8, -2.2] | 0.018 |
| tky | retrieval-jaccard | qwen3b | 500 | 14.6 [11.6, 17.8] | 35.2 [31.0, 39.6] | 0.231 [0.201, 0.262] | 0.252 [0.220, 0.285] | 99.6 | 0.0 | -4.6 [-8.4, -1.0] | 0.183 |
| tky | llmmob | qwen7b | 500 | 16.0 [12.8, 19.2] | 39.8 [35.6, 44.0] | 0.257 [0.225, 0.288] | 0.285 [0.251, 0.318] | 100.0 | 0.0 | -3.2 [-6.4, -0.2] | 0.426 |
| tky | retrieval-jaccard | qwen7b | 500 | 17.0 [13.8, 20.4] | 38.0 [33.8, 42.2] | 0.258 [0.227, 0.291] | 0.280 [0.247, 0.315] | 99.8 | 0.0 | -2.2 [-5.6, +1.0] | 1.000 |

## E2: revisit vs new targets (Acc@1)

Revisit = the target POI appears in the user's history or current trajectory (fixed definition, docs/data_pipeline.md).

| Dataset | Method | Model | Subset | n | LLM Acc@1 | Freq Acc@1 | Δ vs freq | p |
|---|---|---|---|---|---|---|---|---|
| msteps_tokyo | llmmob | qwen3b | revisit | 158 | 25.9 [19.0, 32.9] | 35.4 [27.8, 43.0] | -9.5 [-17.7, -1.3] | 0.688 |
| msteps_tokyo | llmmob | qwen3b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen3b | revisit | 158 | 32.3 [24.7, 39.9] | 35.4 [27.8, 43.0] | -3.2 [-12.0, +6.3] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen3b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| msteps_tokyo | llmmob | qwen7b | revisit | 158 | 40.5 [32.9, 48.1] | 35.4 [27.8, 43.0] | +5.1 [-2.5, +12.7] | 1.000 |
| msteps_tokyo | llmmob | qwen7b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen7b | revisit | 158 | 39.9 [32.3, 47.5] | 35.4 [27.8, 43.0] | +4.4 [-3.8, +13.3] | 1.000 |
| msteps_tokyo | retrieval-jaccard | qwen7b | new | 176 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | llmmob | qwen3b | revisit | 391 | 21.2 [17.1, 25.3] | 27.9 [23.5, 32.2] | -6.6 [-11.8, -1.8] | 0.286 |
| nyc | llmmob | qwen3b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | retrieval-jaccard | qwen3b | revisit | 391 | 22.3 [18.2, 26.3] | 27.9 [23.5, 32.2] | -5.6 [-11.0, -0.5] | 0.905 |
| nyc | retrieval-jaccard | qwen3b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | llmmob | qwen7b | revisit | 391 | 29.7 [25.3, 34.3] | 27.9 [23.5, 32.2] | +1.8 [-2.8, +6.4] | 1.000 |
| nyc | llmmob | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | revisit | 391 | 29.2 [24.8, 33.8] | 27.9 [23.5, 32.2] | +1.3 [-4.1, +6.6] | 1.000 |
| nyc | retrieval-jaccard | qwen7b | new | 109 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| tky | llmmob | qwen3b | revisit | 361 | 18.8 [15.0, 23.0] | 26.6 [22.2, 31.0] | -7.8 [-12.5, -3.3] | 0.035 |
| tky | llmmob | qwen3b | new | 139 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| tky | retrieval-jaccard | qwen3b | revisit | 361 | 20.2 [16.3, 24.4] | 26.6 [22.2, 31.0] | -6.4 [-11.4, -1.4] | 0.404 |
| tky | retrieval-jaccard | qwen3b | new | 139 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| tky | llmmob | qwen7b | revisit | 361 | 22.2 [18.0, 26.6] | 26.6 [22.2, 31.0] | -4.4 [-8.9, +0.0] | 1.000 |
| tky | llmmob | qwen7b | new | 139 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |
| tky | retrieval-jaccard | qwen7b | revisit | 361 | 23.5 [19.1, 28.0] | 26.6 [22.2, 31.0] | -3.0 [-7.8, +1.7] | 1.000 |
| tky | retrieval-jaccard | qwen7b | new | 139 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 1.000 |

## E3: venue-name test (Massive-STEPS)

| Model | Method | Condition | n | Acc@1 | Acc@5 | MRR | Copy | Invalid |
|---|---|---|---|---|---|---|---|---|
| qwen3b | llmmob | aliased | 334 | 12.9 [9.6, 16.5] | 36.2 [31.1, 41.3] | 0.230 [0.194, 0.268] | 97.6 | 0.0 |
| qwen3b | llmmob | present | 334 | 12.3 [9.0, 15.9] | 35.3 [30.2, 40.7] | 0.216 [0.181, 0.253] | 99.1 | 0.0 |
| qwen3b | llmmob | removed | 334 | 12.0 [8.7, 15.6] | 33.2 [28.1, 38.3] | 0.211 [0.175, 0.248] | 95.5 | 0.3 |
| qwen3b | retrieval-jaccard | aliased | 334 | 15.6 [11.7, 19.5] | 33.5 [28.4, 38.6] | 0.235 [0.197, 0.275] | 94.3 | 0.0 |
| qwen3b | retrieval-jaccard | present | 334 | 15.3 [11.4, 19.2] | 32.0 [26.9, 37.1] | 0.225 [0.187, 0.265] | 90.4 | 0.0 |
| qwen3b | retrieval-jaccard | removed | 334 | 15.3 [11.7, 19.2] | 31.7 [26.6, 36.8] | 0.221 [0.183, 0.261] | 90.1 | 0.0 |
| qwen7b | llmmob | aliased | 334 | 20.4 [16.2, 24.9] | 37.1 [32.0, 42.5] | 0.276 [0.233, 0.319] | 100.0 | 0.0 |
| qwen7b | llmmob | present | 334 | 19.2 [15.0, 23.7] | 37.1 [32.0, 42.2] | 0.267 [0.227, 0.310] | 100.0 | 0.0 |
| qwen7b | llmmob | removed | 334 | 20.1 [15.9, 24.6] | 37.4 [32.3, 42.8] | 0.276 [0.234, 0.319] | 100.0 | 0.0 |
| qwen7b | retrieval-jaccard | aliased | 334 | 20.4 [16.2, 24.6] | 34.4 [29.3, 39.5] | 0.263 [0.221, 0.306] | 99.7 | 0.0 |
| qwen7b | retrieval-jaccard | present | 334 | 18.9 [14.7, 23.1] | 32.9 [28.1, 38.0] | 0.247 [0.206, 0.289] | 99.1 | 0.0 |
| qwen7b | retrieval-jaccard | removed | 334 | 19.8 [15.6, 24.0] | 34.4 [29.3, 39.5] | 0.256 [0.214, 0.299] | 99.1 | 0.0 |

| Model | Method | Comparison | n | Δ Acc@1 | Δ Acc@5 | p (Acc@1) |
|---|---|---|---|---|---|---|
| qwen3b | llmmob | present − aliased | 334 | -0.6 [-2.7, +1.5] | -0.9 [-3.6, +1.8] | 1.000 |
| qwen3b | llmmob | present − removed | 334 | +0.3 [-2.4, +2.7] | +2.1 [-0.9, +5.1] | 1.000 |
| qwen3b | llmmob | aliased − removed | 334 | +0.9 [-1.5, +3.3] | +3.0 [+0.3, +5.7] | 1.000 |
| qwen3b | retrieval-jaccard | present − aliased | 334 | -0.3 [-2.4, +1.5] | -1.5 [-3.9, +1.2] | 1.000 |
| qwen3b | retrieval-jaccard | present − removed | 334 | +0.0 [-2.7, +2.7] | +0.3 [-2.4, +3.0] | 1.000 |
| qwen3b | retrieval-jaccard | aliased − removed | 334 | +0.3 [-2.1, +2.7] | +1.8 [-0.6, +4.2] | 1.000 |
| qwen7b | llmmob | present − aliased | 334 | -1.2 [-3.6, +1.2] | +0.0 [-2.1, +2.1] | 1.000 |
| qwen7b | llmmob | present − removed | 334 | -0.9 [-3.6, +1.8] | -0.3 [-2.4, +1.8] | 1.000 |
| qwen7b | llmmob | aliased − removed | 334 | +0.3 [-1.5, +2.1] | -0.3 [-2.4, +1.8] | 1.000 |
| qwen7b | retrieval-jaccard | present − aliased | 334 | -1.5 [-3.9, +0.9] | -1.5 [-3.9, +0.9] | 1.000 |
| qwen7b | retrieval-jaccard | present − removed | 334 | -0.9 [-3.3, +1.5] | -1.5 [-4.2, +1.2] | 1.000 |
| qwen7b | retrieval-jaccard | aliased − removed | 334 | +0.6 [-1.8, +3.0] | +0.0 [-2.7, +2.7] | 1.000 |

## Model size: 7B − 3B

| Dataset | Method | Condition | n | Δ Acc@1 | Δ Acc@5 | p |
|---|---|---|---|---|---|---|
| msteps_tokyo | llmmob | aliased | 334 | +7.5 [+3.6, +11.7] | +0.9 [-2.7, +4.2] | 0.005 |
| msteps_tokyo | llmmob | present | 334 | +6.9 [+3.3, +10.8] | +1.8 [-1.2, +4.8] | 0.005 |
| msteps_tokyo | llmmob | removed | 334 | +8.1 [+4.5, +12.0] | +4.2 [+0.9, +7.5] | <0.001 |
| msteps_tokyo | retrieval-jaccard | aliased | 334 | +4.8 [+1.2, +8.4] | +0.9 [-2.1, +4.2] | 0.062 |
| msteps_tokyo | retrieval-jaccard | present | 334 | +3.6 [+0.0, +7.2] | +0.9 [-2.1, +3.9] | 0.200 |
| msteps_tokyo | retrieval-jaccard | removed | 334 | +4.5 [+0.9, +8.1] | +2.7 [-0.3, +5.7] | 0.085 |
| nyc | llmmob | none | 500 | +6.6 [+3.0, +10.2] | +6.2 [+2.8, +9.8] | 0.005 |
| nyc | retrieval-jaccard | none | 500 | +5.4 [+1.8, +9.0] | +1.4 [-1.4, +4.2] | 0.029 |
| tky | llmmob | none | 500 | +2.4 [-0.6, +5.4] | +3.8 [+0.6, +7.2] | 0.311 |
| tky | retrieval-jaccard | none | 500 | +2.4 [-0.6, +5.4] | +2.8 [+0.0, +5.6] | 0.311 |
