# Complete WarBias results

This saved run contains all six models and three methods: 18 actual training arms, 18 training-source pairs, 216 held-out native pairs and 324 metric entries. All three notebooks executed nine code cells and produced seven figures each. The 24 CSV files preserve the saved values, intervals, units and missing dependencies.

| Method | Executed notebook | Held-out table | Training-source table |
| --- | --- | --- | --- |
| RMU | [Notebook](completed_bundle_v1/notebooks/executed/warbias_rmu_results.ipynb) | [CSV](completed_bundle_v1/tables/rmu_heldout.csv) | [CSV](completed_bundle_v1/tables/rmu_training_source.csv) |
| Task vector | [Notebook](completed_bundle_v1/notebooks/executed/warbias_task_vector_results.ipynb) | [CSV](completed_bundle_v1/tables/task_vector_heldout.csv) | [CSV](completed_bundle_v1/tables/task_vector_training_source.csv) |
| WGA+KL | [Notebook](completed_bundle_v1/notebooks/executed/warbias_wga_kl_results.ipynb) | [CSV](completed_bundle_v1/tables/wga_kl_heldout.csv) | [CSV](completed_bundle_v1/tables/wga_kl_training_source.csv) |

[All tables](completed_bundle_v1/tables), [complete archive](completed_bundle_v1.tar.gz), [publication checksums](completed_bundle_v1_publication.json).

The review folder contains 53 unchanged original files, including all eighteen recorded metric reports. It is a partial materialization for reading the notebooks and CSVs. The complete archive contains all 820 files, including the original raw input records. The archive is about 38 MB; its extracted contents use about 879 MB. Full verification and replay require extraction first.

From this reports directory, extract a disposable copy outside the checkout:

```sh
mkdir /tmp/warbias-report-replay
tar -xzf completed_bundle_v1.tar.gz -C /tmp/warbias-report-replay
PYTHONDONTWRITEBYTECODE=1 python completed_bundle_v1_verify.py \
  /tmp/warbias-report-replay \
  --checksums-sha256 cf72207fafcfc10499791f32bdb36e66458cae138e3dba5d91dc57ab4b3651d7
```

Use a fresh directory name if that path already exists. The archive has no outer directory, so do not strip path components. Verify before replay. Re-execution changes notebook output bytes; keep the original archive unchanged.

Replay from the extracted root or a notebook directory beneath it. The closest ancestor containing `scripts/unlearning_report_data.py` defines the snapshot root. The default manifest is `evaluation/unlearning/warbias_primary_20261031/reports/manifest_completed.json`. The report kernel needs nbformat, nbclient, pandas and Matplotlib.

Metric dependencies remain explicit. Missing calibration, reviewed labels, statistical support and source views remain unavailable or partial. Task-vector scale zero remains a null intervention. The recorded comparisons do not establish successful mitigation on every model.

The maintained [RMU](../../../../notebooks/executed/warbias_rmu_results.ipynb), [task-vector](../../../../notebooks/executed/warbias_task_vector_results.ipynb) and [WGA+KL](../../../../notebooks/executed/warbias_wga_kl_results.ipynb) notebooks contain the same complete data with the current method index and simple presentation. Their [execution check](canonical_report_verification_v1.json) binds the current loader, builder and manifest.

For their default manifest, extract the archive into `completed_bundle_v1` from this reports directory. To use the disposable copy above instead, set `FAIRFORGET_REPORT_MANIFEST` to `/tmp/warbias-report-replay/evaluation/unlearning/warbias_primary_20261031/reports/manifest_completed.json` before replay. Original input records remain unchanged.
