# WarBias training and paired measurement series

The user explicitly selected WarBias for both training and measurement on
31 October 2026. This supersedes the earlier continuation of StereoSet training
for the requested three notebooks. Preserve all completed StereoSet runs and
submitted-job evidence as historical records. The five newly queued StereoSet
training jobs and their five dependent collectors were cancelled before they
started. The five base-model WarBias collection jobs remain valid.

The requested primary arms are fixed-vector RMU, task-vector negation and
WGA+KL, the registered GA-family representative. Use all six pinned models from
the preserved baseline configuration: Qwen3 8B, Llama 3.1 8B Instruct, Gemma 3
12B IT, Lapa 12B Instruct, Mamay 12B v2 and Mamay 27B v2. This series requires
fresh WarBias training for Qwen as well as the other models. A completed
StereoSet intervention cannot fill a WarBias-trained cell.

## Source review and protected families

The source remains published `FairForget/WarBias` revision
`164f355623c2da0028821e0942cad493558a6c13`. Do not replace it with ongoing
unpublished construction outputs. The fixed training-signal bundle has 792
Ukrainian rows across 22 components; development has 257 rows across 15.
Those original bundles and their preparation permissions remain unchanged.

The [cross-view audit](warbias_training_source_audit_20261031.json)
protects all twelve locked WarBias views. It resolves public case IDs, explicit
claim components and parent links. The legacy QA views use scenario clusters;
comparing cluster strings alone misses their relation to training cases.
Protecting those links excludes 655 original training rows and 161 development
rows. The remaining review pool has 137 training rows in 12 cases/9 components
and 96 development rows in 10 cases/9 components. The source audit found no
normalized exact stereotype, counter-stereotype or unrelated sentence matches
between these pools and any locked view.

The [new data record](data/data_manifest.json)
is the authority for final AI eligibility decisions, exact examples, masks and
partitions. It can further reduce the structural pool. The user authorized AI
review and curation; this is not human validation. Keep `human_validated=false`.
Review unsupported war-related group associations rather than treating an
intrinsic attribute, grammatical gender or valid factual statement as a
forget target.

The final review retains all 137 structurally eligible training rows and all 96
safe development rows. The training actor codes are `idp` (60 rows) and
`veteran` (77). Freeze these two provenance-defined actor targets before
collection. M3a may use a genuinely matched held-out actor outside this set;
the code list alone does not establish such a contrast or sufficient support.
All available gender/age profiles are included, so this selection does not
define a separate non-target demographic contrast. No topic target is invented.

Keep original text intact. Bind each stereotype completion to exact source
character spans and an appropriate literal prefix. A counter-stereotype may
have a different prefix; do not rewrite it to manufacture a shared one.
Separate forget, corrective and genuine preservation roles. The three primary
arms do not add corrective CE. Ordinary unrelated WarBias contexts can supply
reference preservation after content review. Target and preservation rows may
share training components, but their union must remain disjoint from all
development and locked components. Export source IDs and stream-specific masks.

## Fixed integration recipes

This is a first WarBias integration series, not a development-tuned or
confirmatory method comparison. Freeze the following settings before final
measurement: seed 42, learning rate 1e-5, batch 1, maximum length 256, gradient
clipping 1 and gradient checkpointing. Use one complete pass over every reviewed
training target, with `epochs=1` and `max_steps` equal to the final row count.
The earlier cap 128 would omit reviewed examples in the 137-row structural
pool. This declared one-pass budget is not selected using final results.
Actual valid loss-token exposure, parameter counts and optimizer steps must
come from the run. Equal passes do not imply equal compute across methods.

WGA+KL retains detached confidence exponent 1 and token-mean base-to-current KL
weight 1 on genuine preservation tokens. Task-vector acquisition retains pure
completion CE and rank 8/alpha 16 q/v LoRA, zero dropout and no bias updates.
Its scales remain 0, 0.25, 0.5, 1 under the existing 0.05-nat retained-token KL
development engineering gate. Scale 0 is a recorded null intervention. This gate
is not a calibrated general capability tolerance.

RMU retains fixed norm 20, hook 7 and `mlp.down_proj.weight` updates in layers 5-7,
with FP32 trainable masters and BF16 computation/reference. Use the verified
text decoder for each architecture. Mamay 27B reserves a separate reference GPU;
other models use their bound one-GPU placement. Do not update vision parameters.

Each model-method arm first runs a separate one-step qualification from the
pinned base. Full training starts only after its model's three qualification
runs pass. Qualification artifacts are not the primary intervention. Full
training reloads the frozen base and uses the frozen complete recipe.

## Before/after outcomes and interpretation

Keep two explicitly separate measurement panels. First, a matched diagnostic
on the final applied training-source IDs measures the model's behavior on its
own target examples. Its native adapted bundle retains `split=train` and
`source_role=training_signal`. Use the maintained Fair-UK `run` and
`paper compare-native` commands with unchanged scoring. These outputs are
training-source diagnostics, not held-out evidence or task-vector selector
inputs.

Second, collect all twelve unchanged locked bilingual WarBias views before and
after each intervention, with the common serial native profile and repeated
development qualification. Since no trained claim component remains in those
views, their intervention effects measure transfer to held-out associations.
Preserve component-macro units, ties, raw predictions, uncertainty and matched
source identities. Never pool the two panels into a single effect.

Every report retains all 18 Tetiana metric entries with their actual dependency
status. The WarBias endpoint package supplies twelve native pairs. Other
uncollected native tasks remain pending; unpublished matched English StereoSet
remains source-policy blocked. Thresholds, independent-support rules, a judge,
practical RI limits and reviewed severity labels remain unset until qualified.
No missing score, failed gate, invalid answer or absent denominator becomes 0.

The three requested executed notebooks must distinguish actual training and
fresh native collection from saved-result replay. Pending jobs do not establish
completed interventions or mitigation. The
[execution record](README.md) binds the
selected releases, jobs and actual results as they become available.
