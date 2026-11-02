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
It assigns complete components to training, development and test with target
fractions of 65%, 15% and 20%. The same global split applies to all four directions.

The [allocator](../src/fairforget/unlearning/split.py) gives 34 components to
training, eight to development and ten to test using largest-remainder rounding.
It uses 32 starts with seed 42 and improving component swaps. It first satisfies
coverage, then reduces per-group fraction deviations for rows, cases and components.
It uses source labels only. Both groups occur in every split. A group/severity
stratum with at least three components occurs in every split; two components
supply training and test, and a singleton stays in training. This search does not claim a globally optimal allocation.

| Group | Training | Development | Test |
| --- | ---: | ---: | ---: |
| IDP | 265 | 62 | 82 |
| Veteran | 389 | 90 | 119 |
| Total | 654 | 152 | 201 |

Counts are triplets per language. Development contains five components per
group, with eight distinct components across both groups. Veteran severity
level 5 has only two components, so it appears in training and test.
Evaluation uses the same 201 test triplets and all 240 baseline QA questions per
language. QA questions never supply gradient data. Some QA claims also occur
in training or development triplets; each bundle records this overlap. The full
QA panel measures performance on the QA format, with shared claims. Whole-release
triplet scores are separate diagnostics.

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
The limits are 530 updates for IDPs and 778 for veterans, equivalent to two
passes over each original forget set. Expanded BiasUnlearn pools can leave the
last pass incomplete.

NPO uses mean token log-likelihood ratios, rather than sequence-sum ratios.
BiasUnlearn adds counter-stereotype cross-entropy and reference KL. Keep its
raw-preference switching rule: above 50%, use the normal pools; below 50%, use
the reversed pools; at 50%, keep the current direction. The controller uses
unweighted development rows and aims to balance raw stereotype preference.
Report both raw and primary neutral-adjusted scores through Fair-UK. The primary
score uses component weights and neutral controls; the switching rule does not
guarantee balance on that score. Keep this WarBias adaptation distinct from an
exact reproduction of the original study.
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
Round the LoRA update to the base dtype before addition, as in the pinned PEFT
safe merge. Verify every resulting parameter and all unchanged parameters.
Record the difference
between adapter and rounded-merge outputs. Reject non-finite outputs or loss of
the observed update. Compare reloaded logits with the verified merged logits at
the same probe positions, using the dtype-specific tolerances in the
[configuration](../configs/unlearning/warbias_final.json). Reject changed tokenizer
inputs or reload drift. Only a verified reload records `trained_merged`.
Keep the exported tokenizer settings unchanged. Training can use masked EOS
padding when the tokenizer has no padding token.
These checks verify checkpoint export, not mitigation quality.

Keep one final adapter per run and the pinned base model. Training uses the fixed
update budget and does not save intermediate checkpoints or select by test
scores. Keep recipes, logs, export checks and all evaluation results.

Rebuild a completed checkpoint, or recover a failed merge, without repeating training:

```bash
python -m fairforget.unlearning.trainer --config /path/to/run/recipe.json \
  --export-from /path/to/run --output /path/to/new-export
```

The recipe and completed step log must match. The new receipt binds the original
adapter and training evidence by hash. It keeps the original run unchanged.
Configuration references may move only when both files retain the recorded hash.
Model paths, training data and method settings must remain unchanged.
Rebuilding a completed export checks the adapter files and requires the same
merged parameters. A temporary merged copy can be removed after matched
evaluation and reconstruction checks pass. Retain its file hashes, tokenizer
and configuration so the saved results remain reproducible.

## Evaluation

Load the pinned base with the trained adapter separately for evaluation.
Verify both file identities. Keep merged exports as separate results.
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
201 held-out triplets and all 240 QA questions in both phases. Save full-release
triplet diagnostics separately. The [native benchmark notebook](native_before_after.ipynb)
retains the other registered integrations. Preparation and CPU checks do not
establish mitigation results.
