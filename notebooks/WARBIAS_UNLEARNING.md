# WarBias unlearning

NPO+KL and BiasUnlearn-style use the final WarBias source. It contains
1,007 aligned triplets per language: 409 IDP and 598 veteran items.

| Training direction | Group | Language |
| --- | --- | --- |
| `idp_uk` | IDP | Ukrainian |
| `idp_en` | IDP | English |
| `veteran_uk` | Veteran | Ukrainian |
| `veteran_en` | Veteran | English |

Each direction trains on one group and one language. All five pinned models
start independently from their base checkpoints. This gives 20 runs per method
and 40 across the two methods. Evaluation covers both groups in both languages.
The application tests whether WarBias supplies useful data for bias mitigation.
Each unchanged base checkpoint is the comparison control. The main study uses
one fixed training seed; uncertainty over cases is reported separately.

## Data and losses

The [data builder](../src/fairforget/unlearning/warbias_data.py) verifies the
source hashes and [component mapping](../warbias-artifacts/evaluation/warbias_baseline/profile_mapping.jsonl).
It protects every component linked to a QA case from training and development.
The allocation selects six of the 28 eligible components for development and
keeps 22 for training. It balances group and severity coverage using source
labels only, then minimizes per-group development fraction deviations for
rows, cases and components. SHA256 resolves ties. Severity strata with only
one eligible component stay in training. The same global split applies to all four directions.

| Group | Training | Development | Held-out |
| --- | ---: | ---: | ---: |
| IDP | 75 | 20 | 314 |
| Veteran | 155 | 40 | 403 |
| Total | 230 | 60 | 717 |

Counts are triplets per language. Each training bundle selects its group's
rows in its training language. Evaluation retains all 717 held-out triplets
and 240 QA questions per language, with both groups represented.
Available IDP severity levels 3, 4 and 5 occur in training and development.
Its only eligible level-2 component stays in training. Veteran levels 2, 3 and 4
occur in both; all veteran level-5 components remain protected for evaluation.

Forget targets are complete S sentences, and correction targets are complete C
sentences. Both use an empty prompt. Token masks cover the complete sentence.
U sentences from the same group and language supply in-domain reference KL.
This measures preservation of WarBias text; general capability needs separate
evidence.

Each bundle records source and stream hashes, item IDs, component assignments,
target group, training language and held-out bilingual rows. A data binding
selects eligible inputs; it does not record a training run. Existing bundles
and checkpoints are never overwritten.

Use the current Git dataset metadata and its declared corrections. The pinned
HF revision binds the unchanged triplet texts. Apply the dataset package's
correction helper to the frozen profile mapping and preserve semantic components.

## Methods and models

The [configuration](../configs/unlearning/warbias_final.json) defines common
optimizer settings and the two objectives. All methods use the same rows
within a direction and the same optimizer-step limit for that direction.
BiasUnlearn builds expanded adversarial pools and can change their direction
using development scores. Equal step limits do not imply equal token exposure.
Each trainer records actual exposure.
WGA+KL remains an [optional experiment](wga_kl.ipynb), outside the 40-run study.

Every [notebook](README.md) prepares and validates the four bundles, runs a
synthetic CPU step, and displays its 20-run plan. Full training starts only when
`RUN_TRAINING=True`. A run ID combines model, group and training language, so
each checkpoint, log and receipt has its own output directory.

## Checkpoint verification

CLI and notebook runs use the same checkpoint validator. It checks every pinned
model and tokenizer asset against the [asset map](../warbias-artifacts/evaluation/warbias_baseline/model_assets.json)
and rejects missing, changed or unpinned loading files. The shared trainer
enforces the package versions in `requirements-unlearning.txt` and records
the actual package builds, CUDA version and GPU identity in its receipt.

Before training, select fixed S, C and U probes from development components.
Verify every merged parameter against the independently computed LoRA update,
rounded to the base dtype. Verify unchanged parameters too. Record the difference
between adapter and rounded-merge outputs. Reject non-finite outputs or loss of
the observed update. Compare reloaded logits with the verified merged logits at
the same probe positions, using the dtype-specific tolerances in the
[configuration](../configs/unlearning/warbias_final.json). Reject changed tokenizer
inputs or reload drift. Only a verified reload records `trained_merged`.
These checks verify checkpoint export, not mitigation quality.

If a completed run failed during merge verification, export its saved adapter
without repeating training:

```bash
python -m fairforget.unlearning.trainer --config /path/to/run/recipe.json \
  --export-from /path/to/run --output /path/to/new-export
```

The recipe and completed step log must match. The new receipt binds the original
adapter and training evidence by hash. It keeps the original run unchanged.

## Evaluation

Use the same held-out IDs before and after training. Keep scores separate by
group and evaluation language. All metrics and uncertainty estimates must
come from Fair-UK. The [baseline notebook](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb) supports
current per-checkpoint triplet and QA scores, including severity slices.
A full-release score includes training and development items and is a
descriptive diagnostic.

Assess mitigation jointly with evidence-based QA correctness. Report
stereotyped answers on ambiguous questions, accuracy with informative evidence,
unknown answers in each condition and invalid outputs. Keep raw triplet
preference and polarity changes visible. Compute paired component intervals
through Fair-UK. Lower raw stereotype preference alone does not establish
successful mitigation.

Capability checks use separate evaluation-only runs before and after the
intervention. They do not supply training, correction or preservation data.
Their benchmark selection is separate from this WarBias application setup.

Use the [before/after notebook](evaluate_before_after.ipynb) for WarBias scores.
It reuses baseline notebook 08's collector, explicit-choice parser, neutral
controls, group/severity summaries and component bootstrap. Score the same
717 held-out triplets and all QA questions in both phases. Save full-release
triplet diagnostics separately. The [native benchmark notebook](native_before_after.ipynb)
retains the other registered integrations. Preparation and CPU checks do not
establish mitigation results.
