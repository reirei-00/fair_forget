# WarBias evaluation before and after unlearning

The [notebook](../notebooks/12_after_intervention.ipynb) uses the same collector and the evaluation library
scorer as [baseline notebook 08](../notebooks/08_warbias_baseline.ipynb).
WarBias selects experiments and labels their before/after reports. The evaluation library
evaluates the supplied WarBias benchmark inputs. This workflow covers all 24
one-epoch interventions: BiasUnlearn and WGA+KL across three models, two groups
and two languages. Reuse three shared base checkpoints.
Evaluate each adapter only on its training group and language. Cross-group
and cross-language evaluation are deferred.
The [native benchmark workflow](../notebooks/13_native_evaluation.ipynb) remains available
with its [own setup](native-evaluation.md).

| Setting | Before and after |
| --- | --- |
| Collection | 409 IDP or 598 veteran triplets, 120 same-group QA questions and matching neutral controls in the training language |
| Main outcomes | Prior-normalised SS, QA accuracy and QA bias |
| Slices | The training group and its case-level severity |
| QA policies | Native `abc_option_text_v2`; analysis `abc_explicit_choice_v3` |
| Weighting | Equal semantic components after within-component averaging |
| Uncertainty | 1,000 component-bootstrap draws, seed 42 |
| Analysis panels | The same 82 IDP or 119 veteran held-out triplets and 120 same-group QA questions before and after |
| Full target-group scores | Separate descriptive diagnostics |

## Environment and paths

Install the bundled evaluator from the artifact root:

```bash
python -m pip install -e ./evaluator
```

Install this library in the notebook environment.
Use a separate Python 3.12 environment with vLLM 0.18.1, Transformers 4.57.6
and NumPy 2.2.6 for GPU collection. Use `VLLM_BATCH_INVARIANT=1` and
`attention_backend="TRITON_ATTN"` in both phases. Qualify each model on the
target GPU. Collect a fresh base with these settings before adapter comparisons.
Configure the existing baseline inputs exactly as in notebook 08:

```bash
export WARBIAS_BASELINE_RELEASE=/path/to/original-release
export WARBIAS_BASELINE_RESULTS=/path/to/original-predictions
export WARBIAS_BASELINE_METADATA_RELEASE=/path/to/current-metadata-release
export WARBIAS_BASELINE_INFERENCE_RELEASE=/path/to/frozen-original-collector
export WARBIAS_BASELINE_NEUTRAL_RELEASE=/path/to/control-release
export WARBIAS_BASELINE_NEUTRAL_RESULTS=/path/to/control-predictions
export WARBIAS_AFTER_ROOT=/path/to/new-post-evaluation
export WARBIAS_TRAINING_ROOT=/path/to/unlearning-runs
export EVALUATOR_ROOT=/path/to/evaluator/src
export EVALUATION_PYTHON=/path/to/evaluation-environment/bin/python
```

Metadata and inference release variables are optional for fresh collections.
Use fresh base and control collections for separate-adapter comparisons. Read
saved merged-model results with their recorded evaluator and source corrections.
The report identifies the input version from its hashes. Saved predictions for an
earlier dataset version remain results for that version. Changed text and its
neutral controls need new inference in both phases.
Post collection uses the compatible control collector when it contains a frozen
source snapshot. `WARBIAS_AFTER_INFERENCE_RELEASE` can select that source
explicitly. The evaluation library verifies unchanged inference and native scoring code.

Run [setup](unlearning-setup.md) first to prepare the four source-bound training bundles.
Adapter directories use `METHOD/MODEL__GROUP_LANGUAGE/intervention_adapter/`.
Load the pinned base with the saved adapter using `representation="adapter"`.
The evaluation library fingerprints the base and adapter files separately. Do not merge the
adapter into BF16 base weights for this evaluation. Collect both phases with
the adapter-capable evaluator; older collector identities remain separate.

## Workflow

1. Enable `RUN_PREPARE` after training. Verify all selected exports before creating
   any releases. Preserve the baseline model options and select the training group
   and language from its original and control inputs without changing their text. The post profile adds the verified adapter path to the same base model.
2. Enable `RUN_AFTER_COLLECTION` on the GPU host. Reuse the baseline runner,
   serial BF16 options, model-specific attention settings and repeated smoke gates.
   Each adapter supplies two original panels and one neutral panel for its training
   group and language. Validated collections can resume; source and runtime changes fail validation.
3. Enable `RUN_REFERENCE_ANALYSIS` to score the three shared baselines with the
   selected parser, controls and target-group held-out IDs. Reuse the matching rows
   from the shared bases for each training group and language.
4. Enable `RUN_AFTER_ANALYSIS` to apply the same scorer to each post checkpoint.
   Check actual before/after runtime, tokenizer, effective inputs and row IDs.
   Compare with the shared base scores for the same group and language. Save full
   target-group and held-out analyses in separate directories for each run.

For a completed subset, pass explicit `(method, run_id)` pairs in
`WarBiasEvaluation(..., selections=[("wga_kl", "gemma3_12b_it__idp_uk")])`.
Only those exports are required. The experiment configuration remains unchanged.

All metric definitions, calculations and CSV exports come from the evaluation library.
The report binds the selected triplet and QA IDs. All native predictions are verified
before selection, including excluded rows. Empty or undefined outcomes remain
explicit. Analysis outputs require new directories and do not replace native files.

The notebook displays target-group scores, severity slices and diagnostics in the same
units as baseline notebook 08. QA bias remains signed and condition-specific;
raw SS, LMS and format checks remain diagnostics. These are matched checkpoint
scores. Paired intervention effects and their intervals use the separate library
intervention analysis.

QA questions are evaluation-only. Their claim overlap with training and development
triplets is recorded in each data binding. Target-group QA scores do not establish unseen-claim
generalization.
