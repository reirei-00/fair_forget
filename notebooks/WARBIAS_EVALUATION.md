# WarBias baseline evaluation

[The notebook](warbias_baseline.ipynb) checks the frozen release,
lists all 24 baseline cells and metric definitions, then analyzes verified saved
runs. It uses `lm_eval.fair_uk.warbias.baseline` for every calculation.

Install Fair-UK in the same Python environment as the notebook kernel. Until
Fair-UK PR #12 is merged, use this pinned library revision:

```sh
python -m pip install "lm_eval @ git+https://github.com/reirei-00/fair-uk.git@694ce2b822c3aa7131d5e527ed4a6ebcd8a59310"
```

Set the release and result locations:

```sh
export WARBIAS_BASELINE_RELEASE=/path/to/frozen-release
export WARBIAS_BASELINE_RESULTS=/path/to/results
jupyter lab notebooks/warbias_baseline.ipynb
```

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

The library reports the recorded severity levels as slices. Prior-normalised
triplet preference needs current-text neutral controls. Before/after outcomes
need paired intervention runs. These requirements do not turn missing values
into zero scores.
