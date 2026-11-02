# WarBias baseline evaluation

The [notebook](../notebooks/08_warbias_baseline.ipynb) defines the metrics and analyzes
saved runs through Fair-UK. [Saved tables and source bindings](../evaluation/warbias_baseline/results/manifest.json)
support inspection without model inference. Use the companion Fair-UK source revision
`8e16737e4ee1e845a64f9f1ce9bb3329d7e32547` in the same environment:

```sh
python -m pip install -e '.[notebooks]'
python -m pip install /path/to/fair-uk
```

The baseline reports prior-normalised triplet SS, QA accuracy and QA bias,
with severity breakdowns. Raw SS, LMS and answer-format checks are diagnostics.
Before/after effects (forget efficacy, NNR, M2/M3 and RI) belong to paired evaluation.

The notebook labels IDPs (ВПО) and veterans separately. Each language contains
409 IDP and 598 veteran triplets, and 120 QA questions per group. In `scores.csv`,
select `scope=actor` with `actor=idp` or `actor=veteran`; `scope=overall` combines
the groups. Severity tables separate IDPs and veterans. Select `scope=actor_severity`
with `actor` and `severity_level` for these slices. All scores and intervals come
from Fair-UK; the notebook only selects and formats them.

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
The notebook shows the `run_model.py` collection command.

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

For existing runs, use their original release. If the analysis library changed,
set `WARBIAS_BASELINE_INFERENCE_RELEASE` to the frozen collector snapshot.
For corrected case/profile labels, prepare a separate release and set
`WARBIAS_BASELINE_METADATA_RELEASE` to it. Fair-UK verifies unchanged inference
inputs before applying declared metadata corrections; original runs stay intact.

The equivalent CLI command is:

```sh
fair-uk-eval warbias baseline \
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


Saved baseline tables are versioned by `source.git_revision` and the input hashes
in their manifest. Use that source revision to reproduce the saved scores.
The current dataset requires new inference for changed text.
