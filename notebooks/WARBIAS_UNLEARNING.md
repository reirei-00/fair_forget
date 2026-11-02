# WarBias unlearning

WGA+KL, NPO+KL and BiasUnlearn-style use the same final WarBias source and
component split. The source contains 1,007 aligned triplets in Ukrainian and
English. Training uses Ukrainian text. English remains available for transfer
evaluation. No English training series is configured.

## Data and losses

The [data builder](../src/fairforget/unlearning/warbias_data.py) verifies the
source hashes and [component mapping](../evaluation/warbias_baseline/profile_mapping.jsonl).
It protects every component linked to a QA evaluation case from training and
development. The remaining 28 components contain 290 triplets. Seeded SHA256
order rounds 20% of those components up to six development components. Both
languages retain the same assignments.

| Split | Components | Triplets per language |
| --- | ---: | ---: |
| Training | 22 | 220 |
| Development | 6 | 70 |
| Held-out | 24 | 717 |

Forget targets are complete S sentences, and correction targets are complete C
sentences. Both use an empty prompt. Token masks cover the complete sentence;
they do not reuse predicate masks from earlier texts. U sentences supply
in-domain reference KL. This measures preservation of WarBias text. It does
not establish general capability retention.

The bundle records source hashes, stream hashes, item IDs, component assignments
and held-out bilingual rows. A data binding selects eligible training inputs;
it does not record a training run. Existing bundles and checkpoints are never
overwritten.

## Methods and models

The [configuration](../configs/unlearning/warbias_final.json) defines common
optimizer settings and the three objectives. Every method uses all six pinned
[models](../configs/unlearning/models.json), the same training rows and the
same optimizer-step limit: 440, from two passes over the 220 training rows.
BiasUnlearn builds expanded adversarial pools and
can change their direction using development scores. Equal step limits do not
imply equal token exposure. Each trainer records actual exposure.

Each [notebook](README.md) prepares and validates the data, runs a synthetic CPU
step, and displays the six-model plan. Full training starts only when
`RUN_TRAINING=True`. Every model saves its own checkpoint, losses and receipts.

## Evaluation

Use the same held-out IDs before and after training in both languages. All
metrics and uncertainty estimates must come from Fair-UK. The
[baseline notebook](warbias_baseline.ipynb) supports current per-checkpoint
triplet and QA scores, including severity slices. A full-release score includes
training and development items and is a descriptive diagnostic.

Final triplet before/after analysis still needs compatible paired bindings and
current-text neutral controls. The historical `evaluate_before_after.ipynb`
workflow does not support these final triplet panels. Preparation and a CPU
example do not establish a completed intervention or measured mitigation.
