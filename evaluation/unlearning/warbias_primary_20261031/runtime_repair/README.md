# Measurement runtime and checkpoint compatibility

All eighteen original WarBias-trained checkpoints are preserved. Measurement
copies for the twelve Gemma-family exports add only `preprocessor_config.json`
and `processor_config.json` from each exact pinned base. The six Qwen/Llama
exports use their original directories. The sidecar and independent verification
bind every original file hash and the two added files. Original weights remain
unchanged, including file permissions. See [inference_views.json](inference_views.json)
and [materialization receipt](materialization_receipt.json).

Fair-UK [PR 9](https://github.com/reirei-00/fair-uk/pull/9) validates these additive
copies and canonicalizes equivalent tokenizer special-token representations.
A string is mapped only to its own matching AddedToken descriptor, with all
descriptor flags retained. Unknown tokens and conflicting descriptors remain
different. Original run identities are preserved.

Gemma's default-fork worker stalled before loading weights. A separate clean
development probe with `VLLM_WORKER_MULTIPROC_METHOD=spawn` exited successfully
and scored both selected triplets. It changed no inference/scoring settings.
The fresh matched collection uses spawn and repeats qualification for every
model/checkpoint/protocol. The probe alone does not establish complete runtime
qualification or measured mitigation.

The unqualified first measurement jobs and four stalled baseline jobs were
cancelled; their partial outputs remain preserved. Training was not repeated.
New measurements use collector v6, runners v3 and the separately pinned Fair-UK
analysis source. Their new submission record selects the resulting evidence.

The notebook runtime is separate from training and evaluation. Its qualification
executed a saved-data notebook and a plot without Torch, vLLM, model weights or
GPU calls. It does not establish model results. `checksums.json` binds the dated
records in this directory.

The saved-analysis/report index repair uses the actual logger contracts: RMU records ordered integer steps 1..N; WGA and task-vector record 0..N-1. Both represent N applied updates. Runners v4 and the new report package validate those exact ranges, including order and uniqueness, without changing raw histories or exposure events. All 249 local regressions and 75 deployed runner checks pass; the deployed check also reads all eighteen actual loss histories. Only pending CPU/report jobs were replaced.

The next saved-export repair preserves the configured virtual environment's
Python executable path. Resolving its symlink had selected a bare interpreter
without the collector's installed package metadata. Analysis release v2 and
runners v5 preserve all dataset, model, collector and scoring bytes. An actual
Llama RMU qualification completed all twelve native exports and analyses, retained
all eighteen metric entries and verified 31,462 unchanged original input hashes.
It made no model calls. The current offline suite passes 268 regressions;
Fair-UK passes 620 tests with one optional-data skip. The dated v5 evidence index
binds the repair, actual qualification, completed-input proofs and new CPU jobs.
