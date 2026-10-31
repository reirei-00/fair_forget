# Saved metrics for WarBias training

This separate driver analyzes the three selected arms - RMU, task-vector and
WGA+KL - for one matrix model. It uses the maintained frozen Fair-UK CLI and
does not change the older five-method driver's requirements.

The selected training release must contain `matrix.json`, the reviewed WarBias
`data_manifest.json` and `training_binding.json`, and the exact model-specific
recipes. The full training execution must include three successful one-step
qualifications and three completed method receipts. The driver checks model and
tokenizer pins, recipe and source hashes, actual step histories, exposure logs,
the published WarBias gradient source and actual checkpoint identities.

The paired collection must contain all 48 full native collections: twelve
WarBias views for the baseline and three post checkpoints. The driver checks
the bound supplemental configuration and unchanged collector, rechecks the 24
saved two-repeat qualification gates without inference, and verifies all full
native receipts, ordered item IDs, source hashes and checkpoint provenance.
Missing files or a failed upstream stage stop analysis.

Run with the qualified baseline Python 3.12 interpreter:

```bash
python "$RUNNER_RELEASE/deploy/unlearning_primary/run_warbias_metrics.py" \
  --training-release "$TRAINING_RELEASE" --model-id "$MODEL_ID" \
  --training "$TRAINING_OUTPUT" --collection "$COLLECTION_OUTPUT" \
  --analysis-source "$ANALYSIS_RELEASE/evaluator" \
  --collector-release "$COLLECTOR_RELEASE" --output "$NEW_ANALYSIS_OUTPUT"
```

`ANALYSIS_RELEASE` is the separately frozen saved-analysis source selected by
the current deployment receipt. Its package commit must match the driver's
`ANALYSIS_COMMIT`, and all package and evaluator checksums must match. The
interpreter repair selects commit `8f5915e89bc35713de32155ef5ebeb16202556e0`.
`COLLECTOR_RELEASE` selects the original frozen collector v6 and its full
evaluation manifest. These two code identities remain distinct. The new
runner release must include exact copies of `supplemental.py`,
`evaluate_checkpoint.py` and `supplemental_checkpoint.sh` bound by the actual
supplemental configuration. Dataset bytes stay in their external frozen release.

For Slurm, use an explicit runner release so the spooled shell script resolves
the correct files:

```bash
sbatch --dependency="afterok:$PAIRED_COLLECTION_JOB" --kill-on-invalid-dep=yes \
  "$RUNNER_RELEASE/deploy/unlearning_primary/warbias_metrics.sbatch" \
  "$RUNNER_RELEASE" "$TRAINING_RELEASE" "$MODEL_ID" \
  "$TRAINING_OUTPUT" "$COLLECTION_OUTPUT" \
  "$ANALYSIS_RELEASE/evaluator" "$COLLECTOR_RELEASE" "$RESULT_PARENT"
```

The wrapper requests one eight-CPU allocation for at most 12 hours and starts
three independent saved-analysis processes. GPU visibility is empty; offline
settings and one math thread per process apply. It submits no other jobs.

The example requires a collection job still known to the scheduler. A completed
job that has left scheduler memory needs a separately bound completion audit
before submitting its saved analysis without that dependency. The audit must
verify successful accounting, all 48 collections, all 36 native pairs and their
exact source and checkpoint identities. A recorded completion proof does not
populate numerical report cells.

The new output contains actual `training_evidence.json` file hashes, generated
method configs, `warbias_metrics_execution.json`, and one
`METHOD/all_metrics.json` with original sibling `analysis.json` and
`validated_pair.json` files per task. This layout supports the strict notebook
loader. The evidence manifest derives hashes from completed files; it invents
no token counts, optimizer steps, archive or training status.

Each report retains all eighteen metric commitments and all eighteen task
statuses: twelve native WarBias tasks available, five other native tasks
pending and English StereoSet blocked by source policy. Actor targets are copied
only from the reviewed manifest's explicit `training_target.actor_values`; when
present, they must exactly match the actors in the forget stream. This describes
data-defined training coverage and introduces no outcome-selected hypothesis.
Without that field, actor targets remain unspecified. Preservation actors do
not define forgetting targets. Other target axes, judge inputs, annotations and
scientific thresholds remain empty or null. Bootstrap
1000 and seed 42 are preserved. Metrics may remain pending or partial even when
all three reports finish. `all_metrics_qualified` stays false; the driver's
completion status means saved reports exist and pass provenance checks.
