# WarBias evaluation before and after unlearning

The [notebook](evaluate_before_after.ipynb) uses the same collector and Fair-UK
scorer as [baseline notebook 08](../warbias-artifacts/notebooks/08_warbias_baseline.ipynb).
FairForget selects experiments and labels their before/after reports. Fair-UK
evaluates the supplied WarBias benchmark inputs. This workflow covers all 40
selected interventions and five shared base checkpoints.
The [native benchmark workflow](native_before_after.ipynb) remains available
with its [own setup](NATIVE_EVALUATION.md).

| Setting | Before and after |
| --- | --- |
| Collection | 1,007 triplets and 240 QA questions per language; the same neutral controls |
| Main outcomes | Prior-normalised SS, QA accuracy and QA bias |
| Slices | IDPs, veterans, both groups and case-level severity |
| QA policies | Native `abc_option_text_v2`; analysis `abc_explicit_choice_v3` |
| Weighting | Equal semantic components after within-component averaging |
| Uncertainty | 1,000 component-bootstrap draws, seed 42 |
| Held-out panel | The same 717 triplet IDs in both phases; all 240 QA questions |
| Full-release scores | Separate descriptive diagnostics |

## Environment and paths

Use Fair-UK revision `8e16737e4ee1e845a64f9f1ce9bb3329d7e32547`:

```bash
git clone https://github.com/reirei-00/fair-uk.git ../fair-uk
git -C ../fair-uk checkout 8e16737e4ee1e845a64f9f1ce9bb3329d7e32547
python -m pip install -e ../fair-uk
```

Install this library in the notebook environment.
Use a separate Python 3.12 environment with vLLM 0.18.1, Transformers 4.57.6
and NumPy 2.2.6 for GPU collection. Keep its runtime unchanged between phases.
Configure the existing baseline inputs exactly as in notebook 08:

```bash
export WARBIAS_BASELINE_RELEASE=/path/to/original-release
export WARBIAS_BASELINE_RESULTS=/path/to/original-predictions
export WARBIAS_BASELINE_METADATA_RELEASE=/path/to/current-metadata-release
export WARBIAS_BASELINE_INFERENCE_RELEASE=/path/to/frozen-original-collector
export WARBIAS_BASELINE_NEUTRAL_RELEASE=/path/to/control-release
export WARBIAS_BASELINE_NEUTRAL_RESULTS=/path/to/control-predictions
export WARBIAS_AFTER_ROOT=/path/to/new-post-evaluation
export FAIRFORGET_TRAINING_ROOT=/path/to/unlearning-runs
export FAIR_UK_ROOT=/path/to/fair-uk
export FAIR_UK_PYTHON=/path/to/evaluation-environment/bin/python
```

Metadata and inference release variables are optional for fresh collections.
For saved runs, retain their frozen collector and declared corrections.
The report identifies the input version from its hashes. Saved predictions for an
earlier dataset version remain results for that version. Changed text and its
neutral controls need new inference in both phases.
Post collection uses the compatible control collector when it contains a frozen
source snapshot. `WARBIAS_AFTER_INFERENCE_RELEASE` can select that source
explicitly. Fair-UK verifies unchanged inference and native scoring code.

Run [setup](SETUP.md) first to prepare the four source-bound training bundles.
Checkpoint directories use `METHOD/MODEL__GROUP_LANGUAGE/merged_checkpoint/`.

## Workflow

1. Enable `RUN_PREPARE` after training. Verify all 40 exports before creating
   any releases. Preserve the baseline model options and copy its original and
   control inputs unchanged. Only the measured checkpoint identity changes.
2. Enable `RUN_AFTER_COLLECTION` on the GPU host. Reuse the baseline runner,
   serial BF16 options, model-specific attention settings and repeated smoke gates.
   Each checkpoint supplies four original and two neutral panels. Validated
   collections can resume; source and runtime changes fail validation.
3. Enable `RUN_REFERENCE_ANALYSIS` to score the five shared baselines with the
   selected parser, controls and held-out IDs.
4. Enable `RUN_AFTER_ANALYSIS` to apply the same scorer to each post checkpoint.
   Check actual before/after runtime, tokenizer, effective inputs and row IDs.
   Save full and held-out analyses in separate directories.

For a completed subset, pass explicit `(method, run_id)` pairs in
`WarBiasEvaluation(..., selections=[("npo_kl", "qwen3_8b__idp_uk")])`.
Only those exports are required. The experiment configuration remains unchanged.

All metric definitions, calculations and CSV exports come from Fair-UK.
The report binds the selected triplet IDs. All native predictions are verified
before selection, including excluded rows. Empty or undefined outcomes remain
explicit. Analysis outputs require new directories and do not replace native files.

The notebook displays group scores, severity slices and diagnostics in the same
units as baseline notebook 08. QA bias remains signed and condition-specific;
raw SS, LMS and format checks remain diagnostics. These are matched checkpoint
scores. Paired intervention effects and their intervals use the separate Fair-UK
intervention analysis.
