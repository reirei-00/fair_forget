# WarBias baseline evaluation

The [notebook](../notebooks/08_warbias_baseline.ipynb) defines the metrics and analyzes
saved runs through the evaluation library. [Saved tables and source bindings](../evaluation/warbias_baseline/results/manifest.json)
support inspection without model inference. The notebook displays these tables by
default; set `RUN_ANALYSIS = True` to score raw predictions. Install the bundled evaluation library
in a separate environment:

```sh
python -m pip install -e '.[notebooks]'
python -m pip install -e ./evaluator
```

The baseline reports prior-normalised triplet SS, QA accuracy and QA bias,
with severity breakdowns. Raw SS, LMS and answer-format checks are diagnostics.
Before/after effects require matched intervention evaluation.

The notebook labels IDPs (ВПО) and veterans separately. Each language contains
409 IDP and 598 veteran triplets, and 120 QA questions per group. In `scores.csv`,
select `scope=actor` with `actor=idp` or `actor=veteran`; `scope=overall` combines
the groups. Severity tables separate IDPs and veterans. Select `scope=actor_severity`
with `actor` and `severity_level` for these slices. All scores and intervals come
from the evaluation library; the notebook only selects and formats them.

Run from this artifact directory:

```sh
python scripts/prepare_warbias_evaluation.py \
  --output ./outputs/warbias-baseline-release \
  --model-root /path/to/models
export WARBIAS_BASELINE_RELEASE="$PWD/outputs/warbias-baseline-release"
export WARBIAS_BASELINE_RESULTS="$PWD/outputs/warbias-baseline-results"
jupyter lab notebooks/08_warbias_baseline.ipynb
```

Preparation verifies the local dataset and QA manifests and writes the evaluation
bundle and collection runner. `--source-dir` selects another artifact directory.
Weights use the directory/revision layout in the
[model registry](../evaluation/warbias_baseline/models.json).
GPU collection requires vLLM 0.18.1 and the
[pinned inference settings](../evaluation/warbias_baseline/model_assets.json).
Run `python "$WARBIAS_BASELINE_RELEASE/run_model.py" --help` for collection options.

Prepare the separate neutral-group control tasks with the reviewed replacements:

```sh
python scripts/prepare_warbias_evaluation.py \
  --output ./outputs/warbias-neutral-release \
  --model-root /path/to/models \
  --neutral evaluation/warbias_baseline/neutral_controls.jsonl
export WARBIAS_BASELINE_NEUTRAL_RELEASE="$PWD/outputs/warbias-neutral-release"
export WARBIAS_BASELINE_NEUTRAL_RESULTS="$PWD/outputs/warbias-neutral-results"
```

The adjacent `neutral_rules.json` binds each target-group replacement. These
controls have assistant review and no human review. Speakers, status predicates
and cross-status attributes stay unchanged. The notebook uses the same collector
for these two tasks and adds primary SS after their complete outputs are verified.

Use the input release bound to each prediction file.
`WARBIAS_BASELINE_INFERENCE_RELEASE` selects its collector snapshot;
`WARBIAS_BASELINE_METADATA_RELEASE` selects declared case/profile corrections.
The library verifies unchanged inference inputs before applying metadata corrections.

The equivalent CLI command is:

```sh
benchmark-eval warbias baseline \
  --release "$WARBIAS_BASELINE_RELEASE" \
  --results "$WARBIAS_BASELINE_RESULTS" \
  --qa-answer-policy abc_explicit_choice_v3 \
  --output ./baseline-summary
```

Add `--inference-release /path/to/collector-release` and
`--metadata-release /path/to/corrected-release` when needed. The notebook
writes its exports to `$WARBIAS_BASELINE_RESULTS/analysis`. To include controls,
add `--neutral-release "$WARBIAS_BASELINE_NEUTRAL_RELEASE"` and
`--neutral-results "$WARBIAS_BASELINE_NEUTRAL_RESULTS"`.

Saved baseline tables are bound to the input content hashes in their manifest.
Recompute predictions when dataset text changes.
