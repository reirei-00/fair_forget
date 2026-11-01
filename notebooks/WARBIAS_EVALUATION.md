# WarBias baseline evaluation

[The notebook](warbias_baseline.ipynb) checks the frozen release,
lists all 24 baseline cells and metric definitions, then analyzes verified saved
runs. It uses `lm_eval.fair_uk.warbias.baseline` for every calculation.

Install this pinned Fair-UK library revision in the same Python environment
as the notebook kernel:

```sh
python -m pip install "lm_eval @ git+https://github.com/reirei-00/fair-uk.git@c777fe03021b03590fd482444005d3ae3902c5da"
```

From the FairForget repository root, prepare the versioned inputs from the
published Hugging Face release:

```sh
python scripts/prepare_warbias_evaluation.py \
  --output ./outputs/warbias-baseline-release \
  --model-root /path/to/models
export WARBIAS_BASELINE_RELEASE="$PWD/outputs/warbias-baseline-release"
export WARBIAS_BASELINE_RESULTS="$PWD/outputs/warbias-baseline-results"
jupyter lab notebooks/warbias_baseline.ipynb
```

The preparation script verifies the pinned source files and writes the bundle,
model profiles, severity bindings and collection runner. It does not download
weights. The model root uses the directory/revision layout in
[the model registry](../configs/unlearning/models.json). Add
`--source-dir /path/to/hf-snapshot` to use a local copy of the pinned files.

For GPU collection, use the [evaluation environment](EVALUATION.md#evaluation-environment).
See [setup](SETUP.md) for model downloads.

Model collection runs separately through the frozen release's `run_model.py`.
The notebook never starts inference. Missing results remain pending. Re-run it
after collecting complete model outputs.

The same analysis is available from the CLI:

```sh
fair-uk-eval warbias baseline \
  --release "$WARBIAS_BASELINE_RELEASE" \
  --results "$WARBIAS_BASELINE_RESULTS" \
  --output ./baseline-summary
```

For saved runs, use their original release directory. If the analysis library
has changed since collection, set
`WARBIAS_BASELINE_INFERENCE_RELEASE` to that frozen collector release. The
library checks its inference, parser and native scorer code before reanalysis;
original run manifests stay unchanged. The CLI accepts the same snapshot with
`--inference-release /path/to/collector-release`.

The library exports `scores.csv`, `evidence_gaps.csv`, `language_contrasts.csv`,
`group_diagnostics.csv` and `availability.csv`. The notebook displays every table.
Exports retain units, support, missing values and uncertainty; `summary.json`
contains the complete report.

The library reports the recorded severity levels as slices. Prior-normalised
triplet preference needs current-text neutral controls. Before/after outcomes
need paired intervention runs. These requirements do not turn missing values
into zero scores.
