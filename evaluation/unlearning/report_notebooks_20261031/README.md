# Six-model method report notebooks

**Historical StereoSet training scope, superseded on 31 October 2026.** The
user subsequently selected WarBias for both training and measurement. These
notebooks preserve the earlier StereoSet gender intervention, its actual Qwen
receipts and its dated submission plan. They do not report WarBias-trained
interventions or complete the newly requested WarBias experiment notebooks.
Those require separately reviewed WarBias training inputs and actual receipts.

These notebooks execute saved-evidence analysis only:

| Method | Executed walkthrough |
| --- | --- |
| Fixed-target RMU | [RMU report](../../../notebooks/executed/uk_primary_rmu_results.ipynb) |
| Task-vector negation | [Task-vector report](../../../notebooks/executed/uk_primary_task_vector_results.ipynb) |
| WGA+KL, the selected GA-family arm | [WGA+KL report](../../../notebooks/executed/uk_primary_wga_kl_results.ipynb) |

The historical snapshot records Qwen training for all three methods: 40 actual
optimizer steps each. Task-vector acquisition selected scale 0, an explicit
null intervention. All six model rows remain visible. The other fifteen
model-method training cells and all eighteen current WarBias before/after
cells are pending in this evidence snapshot. The notebooks' executed cells
do not establish completed six-model experiments.

`manifest_v2.json` preserves the actual submission of fifteen scheduler jobs for the
remaining five models: one baseline, one three-method training job and one
dependent paired-evaluation job per model. Their saved submission snapshot
records PENDING states. The notebook displays these jobs in a separate table;
submitted jobs do not fill training or numeric measurement cells. The public
[submission receipt](../uk_primary_20261031/six_model_warbias/submission.json)
is bound by SHA-256. Existing Qwen training job 2608191 and serial measurement
job 2609355 are recorded separately.

The superseded extension continued the reviewed Ukrainian StereoSet gender
training scope across the remaining models and measures WarBias endpoints.
The existing Qwen forget stream contains four reviewed StereoSet training
families, with two separate development families. WarBias supplies genuine
preservation content and the evaluation endpoint. Human validation remains
pending. A model's staged weights or load-only checks do not establish
training or before/after evaluation.

`source_inventory.json` is an unchanged copy of the dated six-model audit.
Its 96 original/neutral historical baseline records are a separate protocol;
none supplies a missing method post-run. `manifest.json` binds that audit,
all six immutable model pins, actual recipes, receipts, loss logs and observed
exposure summaries by SHA-256. It contains source metadata without raw examples,
model weights or credentials.
The v1 manifest and source inventory remain unchanged. V2 binds its parent
manifest and the submitted plan, and adds the independently audited tokenizer
pins and the preserved reviewed training-binding hash.

The loader checks completed method status, actual method/model/tokenizer pins, forget-stream identity,
optimizer steps, archived training hashes, checkpoint lineage, full locked
WarBias IDs, source/protocol identity, study phase and before/after execution
settings. A complete available metric report must retain all eighteen metric
entries. Missing calibration and reviewed labels remain pending or partial.
Development checks and preflight-only reports cannot fill evaluation cells.
Task-vector selection must belong to the frozen recipe's candidate scales.

To reproduce this historical saved-evidence replay from the repository root:

```bash
python scripts/build_unlearning_report_notebooks.py --execute
```

This requires a local notebook kernel with nbformat, nbclient, pandas and
Matplotlib. It starts no training, inference, API calls or scheduler jobs.
Without `--execute`, source notebooks go to `notebooks/not_executed`.
The default replay selects `manifest_v2.json`; `--manifest manifest.json` with
the full path explicitly selects the preserved earlier snapshot.

To create a later submission snapshot without overwriting its parent:

```bash
python scripts/build_unlearning_report_notebooks.py \
  --manifest /absolute/path/to/parent-manifest.json \
  --schedule /absolute/path/to/actual-submission.json \
  --new-manifest /absolute/path/to/new-manifest.json --execute
```

When new method runs finish, create a new external report manifest. Each
model-method entry can supply `analysis: {path, sha256}` for the maintained
`all_metrics.json`. Keep its sibling task directories containing the original
`analysis.json` and `validated_pair.json` files. The loader checks their hashes
from the report. Original absolute provenance paths remain recorded; local
validation reads the copied sibling artifacts without rewriting frozen reports.

```bash
export FAIRFORGET_REPORT_MANIFEST=/absolute/path/to/new-report-inputs.json
python scripts/build_unlearning_report_notebooks.py \
  --manifest "$FAIRFORGET_REPORT_MANIFEST" --execute
```

Update the manifest's actual training references and reviewed scope for any new
model run; never copy Qwen receipts into another model row. Preserve each dated
manifest and execution snapshot. The notebook method sections show objectives,
masks, actual parameter inspection, exposure and complete loss curves, selected
checkpoints, native WarBias coverage/results, eighteen metric dependencies and
unexecuted reproduction commands.

The exact producer scripts used for these historical cached notebooks are retained in [source_snapshot](source_snapshot/checksums.json). Their byte hashes remain distinct from the current shared report code. The active WarBias-trained reports are [here](../warbias_primary_20261031/reports/README.md).
