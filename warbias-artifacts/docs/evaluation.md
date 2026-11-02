# WarBias baseline evaluation

The [notebook](../notebooks/08_warbias_baseline.ipynb) defines the metrics and analyzes
saved runs through Fair-UK. [Saved tables and source bindings](../evaluation/warbias_baseline/results/manifest.json)
support inspection without model inference. Use the companion Fair-UK source revision
`69616a41475769c63192c8e3bbd93f788e2a4fef` in the same environment:

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
the groups. Severity tables combine both groups. All scores and intervals come
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

WB-0926 restores its original PTSD text and currently-serving profile (WB-C082).
WB-0928 keeps its mobility text and disability/short-service profile (WB-C058).
The [restoration record](../protocol/history/ptsd_case_restoration.json) preserves
both text versions and the fresh GPT-6 Astra English translation binding.
Saved baseline scores use the earlier duplicate text for WB-0926. They remain
historical results, with their changed source files preserved under
`evaluation/warbias_baseline/source_snapshot/`. The `result_scope.json` record
binds those files and the current release. Recollect WB-0926 in both languages
before reporting updated triplet results. QA text and saved QA results are unchanged.
