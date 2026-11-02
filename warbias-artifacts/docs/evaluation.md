# WarBias baseline evaluation

The [notebook](../notebooks/warbias_baseline.ipynb) defines the metrics and analyzes
saved runs through Fair-UK. [Saved tables and source bindings](../evaluation/warbias_baseline/results/manifest.json)
support inspection without model inference. Use the companion Fair-UK source revision
`05340be92e0ca2c91f77c01949dc6204e7ca7694` in the same environment:

```sh
python -m pip install -e '.[notebooks]'
python -m pip install /path/to/fair-uk
```

Run from this artifact directory:

```sh
python scripts/prepare_warbias_evaluation.py \
  --output ./outputs/warbias-baseline-release \
  --model-root /path/to/models
export WARBIAS_BASELINE_RELEASE="$PWD/outputs/warbias-baseline-release"
export WARBIAS_BASELINE_RESULTS="$PWD/outputs/warbias-baseline-results"
jupyter lab notebooks/warbias_baseline.ipynb
```

Preparation verifies the local dataset and QA manifests and writes the evaluation
bundle and collection runner. `--source-dir` selects another artifact directory.
Weights use the directory/revision layout in the
[model registry](../evaluation/warbias_baseline/models.json).
GPU collection requires vLLM 0.18.1 and the
[pinned inference settings](../evaluation/warbias_baseline/model_assets.json).
The notebook shows the `run_model.py` collection command.

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
  --output ./baseline-summary
```

Add `--inference-release /path/to/collector-release` and
`--metadata-release /path/to/corrected-release` when needed. The notebook
writes its exports to `$WARBIAS_BASELINE_RESULTS/analysis`.
