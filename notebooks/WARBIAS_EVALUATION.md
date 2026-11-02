# WarBias baseline evaluation

The [notebook](warbias_baseline.ipynb) defines the metrics and analyzes saved runs
through Fair-UK. Install the pinned library in the notebook kernel environment:

```sh
python -m pip install "lm_eval @ git+https://github.com/reirei-00/fair-uk.git@871060c8fb03c6a2a06d8fe5b2467b67d15730e8"
```

From the repository root, prepare the inputs and open the notebook:

```sh
python scripts/prepare_warbias_evaluation.py \
  --output ./outputs/warbias-baseline-release \
  --model-root /path/to/models
export WARBIAS_BASELINE_RELEASE="$PWD/outputs/warbias-baseline-release"
export WARBIAS_BASELINE_RESULTS="$PWD/outputs/warbias-baseline-results"
jupyter lab notebooks/warbias_baseline.ipynb
```

Preparation verifies the pinned Hugging Face files and writes the evaluation
bundle and collection runner. Add `--source-dir /path/to/hf-snapshot` for local
inputs. Weights are separate; use the directory/revision layout in the
[model registry](../configs/unlearning/models.json) and see [setup](SETUP.md).
GPU collection uses `run_model.py`; see the
[evaluation environment](EVALUATION.md#evaluation-environment).

For existing runs, use their original release. If the analysis library changed,
set `WARBIAS_BASELINE_INFERENCE_RELEASE` to the frozen collector snapshot.
Fair-UK checks compatibility before reanalysis and preserves the original run
manifests.

The equivalent CLI command is:

```sh
fair-uk-eval warbias baseline \
  --release "$WARBIAS_BASELINE_RELEASE" \
  --results "$WARBIAS_BASELINE_RESULTS" \
  --output ./baseline-summary
```

Add `--inference-release /path/to/collector-release` when needed. The notebook
writes its exports to `$WARBIAS_BASELINE_RESULTS/analysis`.
