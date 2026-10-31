# Fixed scale-one task vector results

All six saved task vectors were applied at scale 1.0. No scale search or new
acquisition training was performed. Every merged checkpoint has verified changed
weight values. The result set contains 72 held-out WarBias pairs and six separate
training-source diagnostic pairs.

The [executed notebook](../../../notebooks/executed/warbias_task_vector_results.ipynb)
contains nine executed code cells and seven figures. Its code, execution counts
and outputs match the archived JUPITER execution. The current copy adds replay
instructions. [Publication checks](reports/publication.json) bind both copies.

- [Held-out metrics](tables/heldout.csv)
- [Training-source metrics](tables/training_source.csv)
- [Metric inventory](tables/metric_inventory.csv)
- [Training parameters](tables/training_parameters.csv)
- [Recorded acquisition exposure](tables/training_exposure.csv)
- [Model coverage](tables/coverage.csv)
- [Native task coverage](tables/native_task_coverage.csv)
- [Additional WarBias metrics](derived/README.md)
- [Complete source archive](reports/completed_bundle_v1.tar.gz)

On WarBias-UK triplets, LMS decreased for all six models. The largest decrease
was 12.39 percentage points for Lapa. SS moved closer to 50 for Qwen, Gemma and
Lapa, and farther from 50 for Llama and both Mamay models. These directions
describe point estimates. The tables preserve paired confidence intervals.

The full 18-entry inventory keeps missing calibration, labels, judgments and
source views explicit. Additional native diagnostics have their own stated
scope. AI content review does not establish human validation or broad utility
retention. RMU, WGA and historical task-vector artifacts remain unchanged.

Mamay 27B uses matched batch-invariant before/after execution with TRITON_ATTN.
Its original failed repeat qualification is preserved separately. The numerical
mode is recorded in native model identity. The engineering tolerance remains
1e-6 and does not define scientific neutrality.

For replay, extract the archive into a fresh directory outside the checkout:

```sh
mkdir /tmp/fairforget-task-vector-replay
tar -xzf reports/completed_bundle_v1.tar.gz -C /tmp/fairforget-task-vector-replay
```

Open `notebooks/executed/warbias_task_vector_results.ipynb` in the extracted
directory. Its default manifest is
`evaluation/unlearning/task_vector_fixed_one/reports/manifest.json`.
The archive has no outer directory. Its `checksums.json` binds all 666 files.
The report uses nbformat, nbclient, pandas and Matplotlib. It reads saved local
evidence and makes no model or judge requests. Preserve the original archive
when replay changes notebook output bytes.
