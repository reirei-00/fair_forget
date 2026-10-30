# rmu: pilot

RMU (Representation Misdirection for Unlearning, adapted from
[WMDP](https://github.com/centerforaisafety/wmdp)) applied to gender stereotypes from StereoSet.

## Pilot experiment

Unlearn gender stereotypes from **Lapa** (`lapa-12b-instruct`) on JUPITER and compare the
model before and after on BBQ.

- **Forget set:** every gender `stereotype` sentence from StereoSet (intrasentence, test + dev).
- **Retain set:** every gender `anti-stereotype` sentence.
- That is `--no-split`: 1026 sentences on each side, 257 batches of 4.

`--max_num_batches` defaults to 80, so without it only 320 of the 1026 sentences are used.

## Running the pilot, end to end

### 1. Laptop: push your code

```bash
make test
git add -A && git commit -m "..." && git push
```

### 2. JUPITER login node: update your clone

```bash
ssh <user>@login.jupiter.fz-juelich.de
cd /e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/bias-mitigation-<user>
git pull
make env      # first time, or after dependencies in pyproject.toml change
make smoke    # optional if it already passed: 2-min check of GPU + Lapa loading
```

### 3. Submit the unlearning job

```bash
make submit PURPOSE="pilot: RMU on Lapa, no-split" JOB_TIME=00:30:00 \
    CMD="python -m rmu.unlearn --model_name_or_path lapa-12b-instruct --no-split --max_num_batches 257 --verbose"
```

It prints the job ID (e.g. `Submitted job 2275123`) plus the log and output paths.

### 4. Watch it

```bash
make jobs                          # PENDING -> RUNNING -> COMPLETED / FAILED
make watch JOB=<id> LOSSES=1       # live losses; Ctrl+C stops watching, not the job
scancel <id>                       # only if something is wrong (e.g. nan)
```

Healthy run: `unlearn_loss` and `unlearn_cosine_sim` go down, `retain_loss` stays small and
`retain_cosine_sim` stays near 1. Any `nan`: cancel and check the log.

### 5. Check the results

```bash
RUN=/e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/runs/<id>
cat $RUN/job.txt                                        # what ran
column -t -s, $RUN/lapa-*/losses.csv | less             # per-step losses (q to quit)
ls $RUN/lapa-12b-instruct_alpha-100_batches-257_layer-7  # the unlearned model
column -t -s $'\t' /e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/usage_log.tsv
```

Edit the job's `result` / `notes` cells in `usage_log.tsv` with what you found (needed for the
grant report).

### 6. Keep what matters

Scratch deletes files after 90 days. Copy the small files to long-term storage now; copy the
24 GB model only if the evaluation says it's worth keeping:

```bash
RES=/e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/results/pilot-<id>
mkdir -p $RES && cp $RUN/job.txt $RUN/lapa-*/losses.csv $RES/
```

### 7. Evaluate before and after (BBQ)

`make evaluate MODEL=<short name or path>` runs BBQ on any model; only the model argument differs:

- before: `MODEL=lapa-12b-instruct` (baselines already in `results/bbq-eval/`)
- after: `MODEL=$RUN/lapa-12b-instruct_alpha-100_batches-257_layer-7`

To unlearn (adaptive RMU) and evaluate in one job, use `make unlearn-eval MODEL=...`; see
[`docs/unlearn_eval.md`](../../docs/unlearn_eval.md).

## Open issues

- Lapa has 48 layers; the defaults (`--layer_id 7`, `--layer_ids 5,6,7`) come from WMDP's
  32-layer Zephyr and haven't been tuned for Lapa.

## Where results go

Without `--output_dir`, the unlearned model is saved to
`/e/scratch/e-dev-2026d09-075/fairforget-bias-mitigation/runs/<SLURM_JOB_ID>/`. Scratch deletes
files after 90 days: copy anything worth keeping to
`/e/project1/e-dev-2026d09-075/fairforget-bias-mitigation/results/`. Paths are set in
`config/default.yaml`.

## Checking the code on a laptop

```bash
make setup && make test
```

To check the whole pipeline without a GPU, use a tiny random model from the Hugging Face Hub
(downloaded automatically; its losses are meaningless):

```bash
uv run python -m rmu.unlearn --model_name_or_path hf-internal-testing/tiny-random-LlamaForCausalLM \
    --layer_id 1 --layer_ids 0,1 --max_num_batches 3 --output_dir runs/tiny
```

Put laptop path overrides in `config/local.yaml` (copy `config/local.example.yaml`).
