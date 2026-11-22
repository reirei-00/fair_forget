# JUPITER runs on 21 November 2026

The [experiment checklist](2026-11-21/status/CHECKLIST.md) tracks the six models and their English and Ukrainian panels. The [study instructions](../README.md) explain how to prepare the frozen split, train a fresh checkpoint and evaluate it with Fair-UK.

## What ran

[JUPITER’s submission records](2026-11-21/jobs.json) identify 202 Slurm submissions for this suite and its environment setup through job `3046464`. Each entry records the original command, working directory and SHA-256 of the batch script retained by Slurm. An array counts as one submission. The 202 submissions used 29 distinct batch scripts. Of those, 3 already existed byte for byte in the FairForget PR history at commit `39876c0`.

The [source inventory](2026-11-21/sources.json) maps 437 paths in the deployment to their captured contents. It covers runner and environment code, training and evaluation launchers, diagnostic scripts, checklist and retry helpers, and their selected configuration files. Of the 152 distinct files, 61 are byte-identical to files in nine published FairForget commits. Each inventory entry for those files gives the full commit ID and the file path at that commit. The [dated scripts directory](2026-11-21/scripts/) contains the other 91 files. Every source has a SHA-256 digest from the JUPITER capture, and each submitted script has a matching digest in the job records.

Slurm retained two pilot scripts that differ from their last working-directory copies. Job `3044782` used [this Gemma precision check](2026-11-21/scripts/scheduler/3044782/precision-check.sbatch). Job `3045543` used [this first Mamay 27B memory test](2026-11-21/scripts/scheduler/3045543/fit-27b.sbatch). The submission records bind each job to its own script. Helper files are a snapshot of the deployment at collection time. Slurm retains the submitted batch file, while a Python or shell helper loaded during the job can change independently.

The latest Lapa recovery uses the [export verifier deployed for the retries](https://github.com/reirei-00/fair_forget/commit/e0957c58e609a1adaf275bd5463e11be0618cfe8). The 17 initial failures grew to 24 across the same three checkpoints. They occurred during export, before Fair-UK scoring. Each checkpoint has one export followed by eight evaluation retries. The submitted export script requested one hour. The pending jobs were shortened to 30 minutes after submission. [Job mapping and original failure evidence](2026-11-21/status/lapa-failures.json) · [Export and evaluation status](2026-11-21/status/lapa-audit.json)

## Lapa Ukrainian BBQ continuations on 22 November

The normal service class on this project allows 12 hours per job. The two timed-out English-trained checkpoints retain their original prediction files. The [continuation plan](2026-11-22/plan.json) binds the exact files and counts to the interrupted attempts. The gender checkpoint saved 55,936 of 58,492 post-training predictions. A new 12-hour inference job resumes a copy with the original Fair-UK CLI and compares its runtime manifest to the saved one. A separate 12-hour job performs the paired comparison after successful inference. Both race/ethnicity prediction sets were complete, so its continuation runs only the paired comparison with a fresh 12-hour budget. Each comparison uses the registered 1,000 bootstrap draws and seed 42. It publishes through the original receipt writer under the exact original evaluation request hash.

The [submission records](2026-11-22/jobs.json) include the three continuation stages and their dependent checklist refresh. The script verifier restores the additional files under `lapa-bbq-timeout-recovery` in the original run directory. To reproduce the 22 November tracker, apply its [checklist patch](2026-11-22/checklist.patch) to the restored 21 November script in `lapa-mamay-preflight`. The patched script has SHA-256 `f0ec5bac1ab70fab1d49715cf7b7acdc5fd386a2519ade7ca4a02ddb872a298a`. Use `submitted.json` as historical job evidence. Fresh submissions use a separate recovery directory.

### Three timeouts after the Lapa export retries

The [second continuation plan](2026-11-22/export-timeouts/plan.json) records the saved predictions from each interrupted Ukrainian BBQ evaluation. All three baseline runs completed 58,492 predictions. English profession also completed its post-training report and goes directly to comparison. Ukrainian gender resumes from 52,864 predictions and Ukrainian profession resumes from 56,800. Each uses the same Fair-UK settings and a separate 12-hour allocation for each remaining stage.

These checkpoints used the corrected export runner. To recover the original request, the continuation reads that checkpoint’s verified English BBQ request and the verified Ukrainian BBQ dataset settings. It checks the resulting request hash against the interrupted job’s directory and verifies the stored files before resuming. The [six submitted job IDs](2026-11-22/export-timeouts/submitted.json) identify two prediction jobs, three comparisons and the checklist refresh. The [live-run checklist snapshot](2026-11-21/status/CHECKLIST.md) gives the original job and each continuation stage for all five checkpoints.

To reproduce the later tracker from the restored scripts, first apply the earlier 22 November checklist patch described above. Apply the [second checklist patch](2026-11-22/export-timeouts/checklist.patch) in `lapa-mamay-preflight`, and the [export status patch](2026-11-22/export-timeouts/lapa_status.patch) and [audit patch](2026-11-22/export-timeouts/audit.patch) in `lapa-export-recovery`. Use `patch -p1 < /path/to/patch` in each target directory. The export audit retains the original job IDs alongside the replacement job IDs.

## Verify or restore the scripts

Run the verifier from `redistibution-artifacts/ethos-stereoset/` in a FairForget Git checkout with Python 3.10 or later.

```sh
python3 jupiter/verify_sources.py
python3 jupiter/verify_sources.py --restore /path/to/new-directory
```

The verifier reads the pinned files directly from Git and checks every digest before creating the destination. It restores the historical paths relative to the cluster’s runs directory and writes each submitted script to `slurm/<job-id>.sbatch`.

If the checkout lacks the pinned history, fetch the exact commits listed in the inventory once. Run this command from the same study directory with `origin` pointing to FairForget.

```sh
python3 - <<'PYTHON' | xargs git fetch --no-tags origin
import json
from pathlib import Path

commits = {
    source["git_commit"]
    for path in Path("jupiter").glob("*/sources.json")
    for source in json.loads(path.read_text())["sources"].values()
    if "git_commit" in source
}
print(*sorted(commits))
PYTHON
```

Verification and restoration work offline once the Git objects are present. To verify a downloaded source archive, pass `--repository /path/to/fairforget-clone` and use a clone containing those commits.

On 21 November 2026, restoration from the nine published commits and 91 local archive files reproduced all 437 deployment paths from the captured run. JUPITER independently confirmed the batch-script hashes for all 202 submissions. The restored runner also replayed the four-domain, tiny-model experiment with the original Ethos method and Fair-UK. Its 16-row matrix matched the earlier software check byte for byte (SHA-256 `22708bf0abf4c4525bc9c52c0ec4ee82ca20c08f96459a86e9878d8b7bc0bdbf`).

The archived scripts preserve the original cluster paths and scheduler directives. Reproducing training or evaluation requires the model revisions, source releases, environments and frozen data described in the study instructions. Use those instructions for fresh work. The dated submission commands refer to the original cluster allocation and job dependencies.
