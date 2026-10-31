# All saved native training-source pairs - 31 October 2026

Eighteen actual paired reports and six completed diagnostic execution receipts were copied byte-for-byte and verified against source SHA256. The reports cover 137 Ukrainian WarBias training rows, 12 source cases and 9 semantic components. These are training-source diagnostics, not heldout evaluation or task-vector selector inputs.

All values below are read directly from saved `effects` records at `scope=component_macro`. SS/LMS endpoints are percent; ICAT uses score points. Parenthesized changes are the saved after-minus-before values on the same scales. The exact native units and paired 95% CIs remain in `summary.json`; `metric_rows` contains 54 directly extracted records.

| Model | Method | SS before -> after (change) | LMS before -> after (change) | ICAT before -> after (change) |
| --- | --- | --- | --- | --- |
| Qwen3-8B | rmu | 30.41 -> 23.91 (-6.50) | 67.68 -> 41.39 (-26.29) | 41.15 -> 19.79 (-21.37) |
| Qwen3-8B | task_vector | 30.41 -> 30.41 (+0.00) | 67.68 -> 67.68 (+0.00) | 41.15 -> 41.15 (+0.00) |
| Qwen3-8B | wga_kl | 30.41 -> 30.25 (-0.15) | 67.68 -> 68.18 (+0.51) | 41.15 -> 41.25 (+0.10) |
| Llama-3.1-8B-Instruct | rmu | 42.94 -> 37.83 (-5.11) | 79.96 -> 0.00 (-79.96) | 68.66 -> 0.00 (-68.66) |
| Llama-3.1-8B-Instruct | task_vector | 42.94 -> 42.94 (+0.00) | 79.96 -> 79.96 (+0.00) | 68.66 -> 68.66 (+0.00) |
| Llama-3.1-8B-Instruct | wga_kl | 42.94 -> 37.57 (-5.37) | 79.96 -> 75.28 (-4.68) | 68.66 -> 56.57 (-12.10) |
| Gemma-3-12B-IT | rmu | 27.00 -> 17.18 (-9.82) | 75.26 -> 61.94 (-13.32) | 40.64 -> 21.28 (-19.36) |
| Gemma-3-12B-IT | task_vector | 27.00 -> 20.31 (-6.69) | 75.26 -> 68.17 (-7.09) | 40.64 -> 27.70 (-12.94) |
| Gemma-3-12B-IT | wga_kl | 27.00 -> 23.53 (-3.47) | 75.26 -> 70.05 (-5.20) | 40.64 -> 32.97 (-7.67) |
| Lapa-12B | rmu | 59.66 -> 31.47 (-28.18) | 75.95 -> 53.63 (-22.32) | 61.28 -> 33.76 (-27.52) |
| Lapa-12B | task_vector | 59.66 -> 59.66 (+0.00) | 75.95 -> 75.95 (+0.00) | 61.28 -> 61.28 (+0.00) |
| Lapa-12B | wga_kl | 59.66 -> 54.62 (-5.04) | 75.95 -> 70.47 (-5.48) | 61.28 -> 63.96 (+2.68) |
| Mamay-12B-v2 | rmu | 38.89 -> 10.49 (-28.40) | 66.10 -> 53.65 (-12.45) | 51.42 -> 11.26 (-40.16) |
| Mamay-12B-v2 | task_vector | 38.89 -> 37.48 (-1.41) | 66.10 -> 65.78 (-0.32) | 51.42 -> 49.32 (-2.10) |
| Mamay-12B-v2 | wga_kl | 38.89 -> 26.53 (-12.36) | 66.10 -> 59.54 (-6.56) | 51.42 -> 31.59 (-19.82) |
| Mamay-27B-v2 | rmu | 27.22 -> 27.51 (+0.29) | 69.82 -> 71.13 (+1.31) | 38.01 -> 39.14 (+1.13) |
| Mamay-27B-v2 | task_vector | 27.22 -> 27.22 (+0.00) | 69.82 -> 69.82 (+0.00) | 38.01 -> 38.01 (+0.00) |
| Mamay-27B-v2 | wga_kl | 27.22 -> 26.58 (-0.63) | 69.82 -> 66.85 (-2.97) | 38.01 -> 35.54 (-2.47) |

Task-vector selections from actual training receipts:
- Qwen3-8B: scale 0.0; null intervention True.
- Llama-3.1-8B-Instruct: scale 0.0; null intervention True.
- Gemma-3-12B-IT: scale 1.0; null intervention False.
- Lapa-12B: scale 0.0; null intervention True.
- Mamay-12B-v2: scale 0.25; null intervention False.
- Mamay-27B-v2: scale 0.0; null intervention True.

SS is centered at 50, so decreasing an already-below 50 score does not establish mitigation. Native training-source score changes do not establish general utility performance. The stored bootstrap is 1000 paired draws, seed 42, semantic components within stratum. No metric, change or interval was recalculated.

Every pair was checked against its completed diagnostic receipt, the frozen v6 collector identity, pinned baseline path/config hash, the certified post path/full native file map, and the original actual training receipt/checkpoint. These are saved-metadata checks; no checkpoint weight file was rehashed or downloaded.

- [Full-precision values and existing CIs](summary.json)
- [Remote paths and exact copied-file hashes](fetch_receipt.json)
- Original paired reports remain under each model's `paired/{method}/paired.json`.
