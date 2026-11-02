# WarBias unlearning

WGA+KL, NPO+KL and BiasUnlearn-style use the final WarBias source. It contains
1,007 aligned triplets per language: 409 IDP and 598 veteran items.

| Training direction | Group | Language |
| --- | --- | --- |
| `idp_uk` | IDP | Ukrainian |
| `idp_en` | IDP | English |
| `veteran_uk` | Veteran | Ukrainian |
| `veteran_en` | Veteran | English |

Each direction trains on one group and one language. All six pinned models
start independently from their base checkpoints. This gives 24 runs per method
and 72 across the three methods. Evaluation covers both groups in both languages.

## Data and losses

The [data builder](../src/fairforget/unlearning/warbias_data.py) verifies the
source hashes and [component mapping](../evaluation/warbias_baseline/profile_mapping.jsonl).
It protects every component linked to a QA case from training and development.
Seeded SHA256 order assigns six of the 28 eligible components to development;
the remaining 22 train. The same global split applies to all four directions.

| Group | Training | Development | Held-out |
| --- | ---: | ---: | ---: |
| IDP | 53 | 42 | 314 |
| Veteran | 167 | 28 | 403 |
| Total | 220 | 70 | 717 |

Counts are triplets per language. Each training bundle selects its group's
rows in its training language. Evaluation retains all 717 held-out triplets
and 240 QA questions per language, with both groups represented.

Forget targets are complete S sentences, and correction targets are complete C
sentences. Both use an empty prompt. Token masks cover the complete sentence.
U sentences from the same group and language supply in-domain reference KL.
This measures preservation of WarBias text; general capability needs separate
evidence.

Each bundle records source and stream hashes, item IDs, component assignments,
target group, training language and held-out bilingual rows. A data binding
selects eligible inputs; it does not record a training run. Existing bundles
and checkpoints are never overwritten.

## Methods and models

The [configuration](../configs/unlearning/warbias_final.json) defines common
optimizer settings and the three objectives. All methods use the same rows
within a direction and the same optimizer-step limit for that direction.
BiasUnlearn builds expanded adversarial pools and can change their direction
using development scores. Equal step limits do not imply equal token exposure.
Each trainer records actual exposure.

Every [notebook](README.md) prepares and validates the four bundles, runs a
synthetic CPU step, and displays its 24-run plan. Full training starts only when
`RUN_TRAINING=True`. A run ID combines model, group and training language, so
each checkpoint, log and receipt has its own output directory.

## Evaluation

Use the same held-out IDs before and after training. Keep scores separate by
group and evaluation language. All metrics and uncertainty estimates must
come from Fair-UK. The [baseline notebook](warbias_baseline.ipynb) supports
current per-checkpoint triplet and QA scores, including severity slices.
A full-release score includes training and development items and is a
descriptive diagnostic.

Final triplet before/after analysis still needs compatible paired bindings and
current-text neutral controls. The historical `evaluate_before_after.ipynb`
workflow does not support these final triplet panels. Preparation and a CPU
example do not establish a completed intervention or measured mitigation.
