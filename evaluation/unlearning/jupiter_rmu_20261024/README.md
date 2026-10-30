# Recovered JUPITER adaptive RMU pilot, 23–24 October 2026

Recovered on 30 October from the shared `fairforget-bias-mitigation` project.
This package preserves original records; no new model training or inference was
needed. `manifest.json` binds 51 source/evidence files by SHA-256. Predictions are
losslessly gzipped, with original uncompressed hashes also recorded.

| Model | Baseline job | RMU + evaluation job | BBQ accuracy before → after |
| --- | --- | --- | --- |
| Lapa 12B | 2468208 | 2492541 | 79.71 → 81.06% |
| Gemma 3 12B | 2470349 | 2493776 | 85.31 → 85.17% |
| Qwen3 8B | 2470343 | 2493909 | 89.86 → 45.42% |

Run `python evaluation/unlearning/jupiter_rmu_20261024/audit.py` from the repository
root. It uses only the Python standard library. It verifies file hashes, all
34,032 saved predictions, paired item metadata, label validity, 129 training
steps per model, original accuracy/bias metrics and counts. It writes
`verification.json` and `summary.csv`. The executed notebook displays these
results and actual loss curves. `historical_bbq_metrics.py` is an unchanged copy
of the team scorer, not a replacement for the maintained Fair-UK evaluator.

Training used English StereoSet intrasentence gender: shuffled halves, stereotype
forget sentences and anti-stereotype retain sentences. Adaptive RMU used beta=1,
alpha=5, learning rate=0.0005, batch size=4, seed=42, loss_positions=words.
Layers 30–32 were updated for Lapa/Gemma and 15–17 for Qwen. The evaluation uses
5,672 English BBQ Gender_identity examples, equally split between ambiguous and
disambiguated contexts; next-token A/B/C choice scoring, not free-text judging.

The copied source snapshot was retrieved after the experiments. Its commit is
not asserted to be the training-time code identity. Actual job receipts name
the Kopach checkout; the scorer hashes match both retrieved team checkouts.
Usage-log exit=0 entries and complete saved model-mode predictions support
completed evaluation; scheduler accounting returned no rows at recovery.

These are descriptive single-seed historical results with no confidence
intervals, general utility battery or calibrated judge. Training-time immutable
checkpoint/tokenizer/effective-prompt hashes are unavailable. Matching saved item
metadata does not establish all current Fair-UK provenance requirements. Do not
feed these records into the maintained paired CLI by fabricating missing run
manifests. Do not combine them with Ukrainian WarBias metric denominators.
The Qwen run has a major accuracy collapse and is not successful mitigation.
No Mamay unlearning result is confirmed in this recovered evidence.

Original users and commands remain in job receipts and the usage log. October
28–29 results are excluded from the reporting period ending October 25.

## Training notebook and mapping coverage

The [training workflow notebook](../../../notebooks/not_executed/jupiter_rmu_training_workflow.ipynb)
was prepared on 30 October. It calls the preserved trainer after checking model
weights, required data, CUDA and a new output directory. It does not claim an
executed October training-notebook history; those runs used the recorded CLI
jobs. The existing executed notebook replays the actual saved results.

All six prediction files have 16 missing `target_loc` values (8 ambiguous and
8 disambiguated), giving target-mapping coverage 5,656/5,672. The original scorer
includes these examples when counting non-unknown outputs but excludes them
from the biased-answer numerator. Original scores are preserved without a
retroactive mapping change. `verification.json` reports missing-target counts
and their non-unknown predictions separately for every model and phase.
