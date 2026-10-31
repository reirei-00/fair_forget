# WarBias method notebooks

These are the current readiness and saved-evidence reports for **WarBias training and WarBias measurements**.

| Method | Executed report |
| --- | --- |
| Fixed-target RMU | [RMU](../../../../notebooks/executed/warbias_rmu_results.ipynb) |
| Task-vector negation | [Task vector](../../../../notebooks/executed/warbias_task_vector_results.ipynb) |
| WGA+KL, the selected GA-family method | [WGA+KL](../../../../notebooks/executed/warbias_wga_kl_results.ipynb) |

Every report retains all six pinned models and all eighteen metric commitments. All eighteen training arms have completed, as recorded in the [dated completion audit](../training_completion_v1.json). The submission replay keeps training and numeric cells pending until full receipt bundles and native reports are imported. The corrected eighteen measurement jobs are shown separately with their recorded submission time. Executed notebook cells establish that the readiness/reporting code ran; they do not establish applied training steps, model performance or mitigation.

The hash-bound [current manifest](manifest_v5.json), preserving the earlier snapshots, verifies the exact reviewed inputs: 137 training rows from 12 cases in nine components, 96 development rows in nine other components, original predicate masks, the six-tokenizer span audit, model pins and eighteen planned method recipes. One full pass is **137 planned optimizer steps per arm**. Human validation is false. All 13 training components linked to locked legacy views were excluded, including their ordinary preservation text; the complete protected component union is 33.

Three stages remain separate:

1. Actual training, losses, token exposure and selected checkpoint, populated only by complete training receipts.
2. Native before/after diagnostics on the exact 137 original **training-source** triplets. These are not held-out results.
3. Native pairs on twelve frozen UK/EN held-out WarBias views, followed by the complete eighteen-entry metric dependency inventory. Missing judging, human calibration, reviewed labels or statistical support remain explicit.

The completed training audit records task-vector scales: Gemma 1, Mamay 12B 0.25, and 0 for the other four models. A zero scale is a null intervention. Full receipt imports will populate the parameter tables. Historical Qwen scale-zero StereoSet acquisition does not supply a result here. Actor targets are the original source codes `idp` (60 rows) and `veteran` (77 rows). This selection does not define a non-target gender/age contrast.

Reproduce only the report replay from the repository root:

```bash
python scripts/build_unlearning_report_notebooks.py --study warbias \
  --manifest evaluation/unlearning/warbias_primary_20261031/reports/manifest_v5.json --execute
```

The kernel needs nbformat, nbclient, pandas and Matplotlib. No model weights, GPU allocation, API access or judge is used. Without `--execute`, the builder writes source notebooks under `notebooks/not_executed`.

To add actual results, preserve this snapshot and prepare a new external manifest. Each model/method can supply hash-bound `training`, `training_diagnostic: {execution, comparison}`, and `analysis` references. Keep `all_metrics.json` with its sibling task directories and native `analysis.json`/`validated_pair.json`. The loader rejects the old StereoSet gradient source, incomplete receipts, changed source hashes, wrong model/tokenizer/method, diagnostic/evaluation phase confusion and checkpoint-lineage mismatches.

```bash
python scripts/build_unlearning_report_notebooks.py --study warbias \
  --manifest /absolute/path/to/new-manifest.json --execute
```

The historical [StereoSet manifest and notebooks](../../report_notebooks_20261031/README.md) remain separate and unchanged. Their original/v2 manifests and executed notebooks are also preserved in the local `outputs/warbias_unlearning_20261031/historical_stereoset_reports` snapshot. No historical intervention or original/neutral measurement is relabelled as fresh WarBias-trained evidence.


Complete all six saved result panels in a new portable output directory:

```bash
python scripts/assemble_warbias_report.py \
  --base-manifest evaluation/unlearning/warbias_primary_20261031/reports/manifest_v5.json \
  --training-submission evaluation/unlearning/warbias_primary_20261031/training_submission.json \
  --measurement-submission evaluation/unlearning/warbias_primary_20261031/measurement_submission_v4.json \
  --training-release /absolute/path/to/frozen/warbias-training-release \
  --output /absolute/path/to/new-completed-report-bundle --execute \
  --kernel-name fairforget-report-v1
```

This saved-only assembly requires all eighteen actual training arms, all eighteen native training-source comparisons and all 216 held-out native pairs. It preserves eighteen metric entries per arm even when their calibration or support remains unavailable. It copies small native evidence, input data, receipts and report code into a portable tree; it copies no model weights, evaluator source or raw exposure logs. Native provenance and original absolute paths remain unchanged. New relative, hash-bound references locate the copied files. `assembly_execution.json` records completion or failure. Existing output directories are refused.

The explicit kernel option is supported by both tools. Local replay defaults to `python3`; the separately qualified JUPITER reporting runtime uses `fairforget-report-v1`. Notebook replay does not install or modify a shared runtime. A three-notebook completed bundle is published only after the saved evidence gates pass and all requested replay cells execute without error.

Completed assembly also writes spreadsheet-ready UTF-8 CSV files under `tables/` for each method: held-out native values, native training-source comparisons, eighteen-entry metric status, actual exposure, training parameters and model/task coverage. All three shared-loader replays must pass before any CSV is created. These are copies of the existing returned tables, with no new metric calculation or pooling. Missing numeric values stay blank with their statuses; nested native scopes and intervals use JSON cells. CSV row counts and SHA-256 hashes are recorded in `assembly_execution.json`, and every CSV is included in the final `checksums.json`.

The preserved [first automatic report job](report_submission_v1.json), **2610782**, was cancelled while pending, before producing outputs. A new report package reads the corrected method-specific log indexes and the [replacement CPU submission](../measurement_submission_v3.json). The first immutable package remains bound by its [deployment receipt](report_deployment_v1.json). Completed assembly will execute the three completed notebooks and write twenty-four CSV tables only after the full saved-evidence checks pass. The job being queued does not establish completed outputs.

All eighteen native TRAIN-source pairs are also preserved in the [completed training-source snapshot](../training_source_results/README.md), with all fifty-four component-macro SS/LMS/ICAT values and their existing confidence intervals. This dated snapshot does not fill held-out panels or establish mitigation. The completed report assembly will import the full training receipts and metric evidence together.

The preserved [second report job](report_submission_v2.json), **2611159**, was cancelled after its CPU dependency failed during saved export. The current v5 snapshot binds six new CPU analyses, the qualified interpreter repair and independently verified completed Qwen/Llama/Lapa collection inputs. Complete assembly still requires every model's actual reports. Raw logs retain RMU indexes 1..137 and WGA/task-vector indexes 0..136. The repair uses the original trained checkpoints and native collections.

The selected [report job](report_submission_v3.json), **2611634**, uses the
[verified v3 package](report_deployment_v3.json) and waits for the six new CPU
jobs 2611593..2611598. Its submission independently verified all six completed
TRAIN diagnostics and their eighteen original paired-report hashes. The current
readiness notebooks each execute nine cells without errors. Full numerical
notebooks and twenty-four CSV tables remain pending until complete assembly.
