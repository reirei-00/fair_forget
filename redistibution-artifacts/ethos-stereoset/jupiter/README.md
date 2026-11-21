# JUPITER runs on 21 November 2026

The [experiment checklist](2026-11-21/status/CHECKLIST.md) tracks the six models and their English and Ukrainian panels. The [study instructions](../README.md) explain how to prepare the frozen split, train a fresh checkpoint and evaluate it with Fair-UK.

## What ran

[JUPITER’s submission records](2026-11-21/jobs.json) identify 116 Slurm submissions for this suite and its environment setup through 18:26 UTC. Each entry records the original command, working directory and SHA-256 of the batch script retained by Slurm. An array counts as one submission. The 116 submissions used 25 distinct batch scripts. Three of those scripts already existed byte for byte in the FairForget PR history at commit `39876c0`.

The [source inventory](2026-11-21/sources.json) maps 437 paths in the deployment to their captured contents. It covers runner and environment code, training and evaluation launchers, diagnostic scripts, checklist and retry helpers, and their selected configuration files. Its 152 distinct files include 61 that already existed in FairForget history. A `prior_git_blob` identifies the original Git content. SHA-256 identifies the captured bytes independently of Git. Repeated files share one snapshot copy. All archived contents live under the dated directory and can be verified independently of future edits to the study runner.

Slurm retained two pilot scripts that differ from their last working-directory copies. Job `3044782` used [this Gemma precision check](2026-11-21/scripts/scheduler/3044782/precision-check.sbatch). Job `3045543` used [this first Mamay 27B memory test](2026-11-21/scripts/scheduler/3045543/fit-27b.sbatch). The submission records bind each job to its own script. Helper files are a snapshot of the deployment at collection time. Slurm retains the submitted batch file, while a Python or shell helper loaded during the job can change independently.

The latest Lapa recovery uses the [export verifier deployed for the retries](https://github.com/reirei-00/fair_forget/commit/e0957c58e609a1adaf275bd5463e11be0618cfe8). The 17 initial failures grew to 24 across the same three checkpoints. They occurred during export, before Fair-UK scoring. Each checkpoint has one export followed by eight evaluation retries. The submitted export script requested one hour. The pending jobs were shortened to 30 minutes after submission. [Job mapping and original failure evidence](2026-11-21/status/lapa-failures.json) · [Export and evaluation status](2026-11-21/status/lapa-audit.json)

## Verify or restore the scripts

Run these commands from `redistibution-artifacts/ethos-stereoset/` using Python 3.10 or later.

```sh
python3 jupiter/verify_sources.py
python3 jupiter/verify_sources.py --restore /path/to/new-directory
```

The second command verifies the contents and writes them under their historical paths relative to the cluster’s runs directory. It also writes the exact submitted script for each job under `slurm/<job-id>.sbatch`. Choose a new destination. Both commands work offline.

The archived scripts preserve the original cluster paths and scheduler directives. Reproducing training or evaluation requires the model revisions, source releases, environments and frozen data described in the study instructions. Use those instructions for fresh work. The dated submission commands refer to the original cluster allocation and job dependencies.
