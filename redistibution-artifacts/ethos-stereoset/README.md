# Single-domain Ethos experiments

The [JUPITER run record](jupiter/README.md) contains the submitted scripts, their checksums and the dated experiment checklist.

Each seed covers 48 independent checkpoints across six base model revisions, four StereoSet domains (gender, profession, race/ethnicity and religion) and two training languages. Both evaluation languages apply to every checkpoint.

The original full matrix runs 24 English jobs before its Ukrainian phase. The following model-specific BF16 configurations each run four English jobs before four Ukrainian jobs. Export the selector before preparing inputs, submitting training or evaluating. Give each configuration its own suite directory. Pass `--after-ok PILOT_JOB_ID` to `suite.py submit` to have Slurm enforce the pilot condition while the pilot is queued or running. All use the same seed-42 split, a learning rate of 0.0005 on both branches and separate checkpoint paths. For Lapa and Mamay 12B, first qualify the model with an English gender pilot. Check all 20 stereotype and two anti-stereotype epoch losses for finiteness and both adapters for finite values and a nonzero LoRA update.

| Model | Selector | Configuration |
|---|---|---|
| Gemma 3 12B IT | `export ETHOS_STUDY=gemma_bf16` | [Gemma](config/stereoset_gemma_bf16.json) |
| Lapa 12B Instruct | `export ETHOS_STUDY=lapa_bf16` | [Lapa](config/stereoset_lapa_bf16.json) |
| Mamay 12B IT v2 | `export ETHOS_STUDY=mamay12_bf16` | [Mamay 12B](config/stereoset_mamay12_bf16.json) |
| Mamay 27B IT v2 | `export ETHOS_STUDY=mamay27_bf16` | [Mamay 27B](config/stereoset_mamay27_bf16.json) |

For Mamay 27B, qualify the full English gender pilot on one GH200 before releasing the eight jobs. Install [the pinned package overlay](ethos/requirements-mamay27-overlay.txt) with `uv pip install --python "$ETHOS_PYTHON" --target "$ETHOS_PACKAGE_OVERLAY" --no-deps -r ethos/requirements-mamay27-overlay.txt` on top of the registered training environment. The separate overlay lets the unchanged trainer load native BF16 text weights. For weights under a `language_model.` prefix, `python ethos/text_view_27b.py REGISTERED_VIEW OUTPUT_VIEW` copies each text tensor under its causal-model name. Run `python ethos/check_27b_base.py OUTPUT_VIEW BASE_JSON REGISTERED_VIEW` in the overlay environment to compare all text parameters with the original weights.

From the overlay environment run `python ethos/mamay_pilot.py --suite "$SUITE" --source "$WORK/Ethos" --registered-view "$REGISTERED_VIEW" --view "$OUTPUT_VIEW" --output "$PILOT" --gpus 1`. For a diagnosed memory failure, request two GPUs and use a fresh output directory with `--gpus 2`. The two-device diagnostic uses PyTorch distributed data parallelism, a batch of three per device, and the original training and SVD programs.

The pilot saves its two native adapters and logs, base.json, and the verified SVD and reload record fit-checkpoint.json. Use `qualify_mamay.py tokens REGISTERED_VIEW SUITE` in the registered environment, then `qualify_mamay.py qualify --help` for the production qualification command in the overlay environment. Export the path to its JSON as `ETHOS_QUALIFICATION` and the package directory as `ETHOS_PACKAGE_OVERLAY` when submitting. A successful qualification binds the pilot, split, token IDs, view files and training recipe. A GPU memory failure triggers a separate two-GPU diagnostic with three examples per device. The one-GPU production profile always requires its own passing pilot.

The wrapper runs the original [Ethos bias trainer and SVD](https://github.com/leigao97/Ethos/tree/88a932f0ff5cd30ad75be163fad9b359e0a7c59d). Configuration follows the authors’ GPT-2 bias recipe with 20 stereotype epochs, two anti-stereotype epochs and SVD scale 2.0. The 128-token block is a study choice for short domain corpora. The upstream trainer drops incomplete blocks at each 1,000-row boundary. The run receipt records complete blocks and discarded tokens for both branches. The model registry pins Llama 3.1 8B Instruct, Gemma 3 12B IT, Qwen3-8B, Lapa 12B Instruct and Mamay 12B/27B IT v2.

For a Ukrainian rerun after all four English checkpoints for that model have verified receipts, `suite.py train --english-gate model` uses the original suite and checkpoint identities. The study's full-array submission retains its whole-suite English prerequisite. On JUPITER the optional eighth positional argument of `train_suite.sbatch` selects the gate. For Llama and Qwen the Ukrainian recovery uses the original FP16 configuration.

Use `suite.py export --suite "$SUITE" --runs "$RUNS" --model "$MODEL" --language "$LANGUAGE" --domain "$DOMAIN"` once per checkpoint before launching evaluation panels concurrently. The export verifies the native checkpoint and its published before/after weights. Schedule each panel after the export for its matching checkpoint. Repeated export calls reuse the verified files.

The export receipt records the FP32 development-logit differences at merge and at save/reload separately. For a native unit-scale Ethos SVD checkpoint that exceeds the adapter-to-export tolerance, the verifier requires identical development token IDs and exactly matching pre-save and reloaded logits. It also checks the identity factor, every reloaded target weight against the correctly rounded sum of the original weight and archived update, and every other state tensor against the original base. Any discrepancy in those checks rejects the export. The receipt records the verifier hash and tensor counts under `dense_svd_exact`.

## Prepare the software

Run the examples in Bash from this directory. Set `WORK` to an absolute, durable directory outside the checkout. Preparation and the design tests use Python 3.10 or newer. Training requires Linux, PyTorch 2.9.1 and the pinned training stack. The model snapshots, frozen input release and results live outside Git.

```bash
export WORK=/absolute/path/to/ethos-study
mkdir -p "$WORK"
git clone https://github.com/leigao97/Ethos.git "$WORK/Ethos"
git -C "$WORK/Ethos" checkout --detach 88a932f0ff5cd30ad75be163fad9b359e0a7c59d
git clone https://github.com/reirei-00/fair-uk.git "$WORK/fair-uk"
git -C "$WORK/fair-uk" checkout --detach db435f59ec65890969358af15084737e2e64bf77
python3 ethos/suite.py plan --seed 42
python3 -m unittest discover -s tests
```

On JUPITER, `bash ethos/setup.sh` provisions one environment per model with the Stages/2026 PyTorch module and [package pins](ethos/requirements-jupiter.txt). Set `ETHOS_PROJECT_ROOT` to override the project storage root and `ETHOS_UV` to override the uv executable. Source `ethos/use.sh MODEL_ID` to select the matching interpreter. On another Linux host use a separate virtual environment with a compatible PyTorch 2.9.1 build and install the pinned packages with normal dependency resolution. Install Fair-UK in a separate virtual environment using its `.[fair-uk]` extra, with a compatible Transformers version for all selected models (4.57.6 in the training environments). Set `EVAL_PYTHON` to its absolute interpreter path. The evaluator records its effective dependency versions and ignores the training interpreter’s Python import path. Set `FAIR_UK_PYTHONPATH` only if the evaluation environment needs extra module directories.

Acquire the exact [model revisions](config/models.json) on a download host with access to the model repositories. From an environment with huggingface-hub installed:

```bash
python - "$WORK/models" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
for m in json.loads(Path("config/models.json").read_text())["models"]:
    snapshot_download(m["repository"], revision=m["revision"],
                      local_dir=Path(sys.argv[1])/m["directory"]/m["revision"],
                      allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja","*.tiktoken"])
PY
```

## Freeze the bilingual split

Download the three pinned sources and reproduce the English files. The conversion verifies the bytes of all inputs and both English outputs. This reconstructs the staged English StereoSet views. English benchmark scoring also needs a registered evaluation release.

```bash
cd "$WORK"
curl -fL https://raw.githubusercontent.com/moinnadeem/StereoSet/ead7d086a64a192a1eca88e0dd2fd163de375218/data/dev.json -o dev.json
curl -fL https://huggingface.co/datasets/FairForget/StereoSet-UK-Unlearning/resolve/8810fc35680b398521c7264dab2c512b5ced7f30/data/train.csv -o uk_pool.csv
curl -fL https://huggingface.co/datasets/FairForget/StereoSet-UK-Eval/resolve/a2ded63f2640e9cf95befd331c852e3184be2f5b/data/validation.csv -o uk_eval.csv
cd -
python3 ethos/source_data.py --upstream "$WORK/dev.json" --uk-pool "$WORK/uk_pool.csv" \
  --uk-eval "$WORK/uk_eval.csv" --output "$WORK/english"
sources=(--uk-pool "$WORK/uk_pool.csv" --uk-eval "$WORK/uk_eval.csv"
         --en-pool "$WORK/english/en_pool.csv" --en-eval "$WORK/english/en_eval.csv")
python3 ethos/suite.py review-template "${sources[@]}" --output "$WORK/review-template.csv"
```

The [executed split notebook](split/stereoset_split.ipynb) documents the full-pool release at seed 42. Open it to inspect the 2,101 source IDs, held-out families, exclusions, counts and leakage checks. Its default path replays the committed evidence with the Python standard library and Jupyter. An automated screen can miss semantic overlap.

To materialize this exact release after creating the bilingual template:

```bash
python3 split/partition.py --template "$WORK/review-template.csv" --scores split/release/scores.csv.gz \
  --spans split/release/claim_links.csv.gz --output "$WORK/allocation"
cmp "$WORK/allocation/assignments.csv" split/release/assignments.csv
python3 ethos/suite.py prepare "${sources[@]}" --review "$WORK/allocation/review.csv" --seed 42 --output "$WORK/suite"
```

To regenerate the similarity evidence from the source template, install [the split dependencies](split/requirements.txt) in a separate environment and use new output paths. Both encoders load exact Hub revisions. The reference score release was generated on Apple MPS. Cross-device floating-point differences can affect a rounded threshold decision. The distributed pair evidence fixes the study run.

```bash
python split/screen.py --template "$WORK/review-template.csv" --output "$WORK/scores.csv.gz" --device cpu
python split/claim_spans.py --template "$WORK/review-template.csv" --output "$WORK/claim_links.csv.gz" --device cpu
```

For a separately reviewed future release, the template contains 2,101 aligned IDs. Its 949 protected evaluation IDs leave 1,152 candidates for training, development or exclusion. Copy the template to `review.csv` and complete the semantic-family and split decisions. Use the same family for related examples across languages and domains. Each retained family belongs to one split. Preserve all protected IDs, and allocate training and development families to all four domains. The suite requires the reviewed split as an input.

```bash
python3 ethos/suite.py prepare "${sources[@]}" --review "$WORK/review.csv" \
  --seed 42 --output "$WORK/suite"
```

## Train and preserve checkpoints

From the checkout available to the compute nodes, submit the two dependent arrays on JUPITER:

```bash
common=(--suite "$WORK/suite" --runs "$WORK/runs")
training=(--source "$WORK/Ethos" --model-root "$WORK/models" --views "$WORK/views")
python3 ethos/suite.py submit "${common[@]}" "${training[@]}"
```

A single allocated GPU can run one cell through the same entry. For example:

```bash
source ethos/use.sh llama31_8b_instruct
cell=(--model llama31_8b_instruct --language en --domain gender)
"$ETHOS_PYTHON" ethos/suite.py train "${common[@]}" "${training[@]}" "${cell[@]}"
```

The trainer reads the stereotype and anti-stereotype columns from the training-only file named `test.csv`, the split name expected by Ethos. For Gemma-based models the training view selects the original text backbone. Each attempt keeps both native adapters, their tensor-equivalent binary exports, the SVD checkpoint, commands, dependency versions, frozen-input identity, base-file hashes and logs. The runner verifies the base weight shards against the Hub revision, matches the reloaded adapter tensors to the SVD output, and reloads the final adapter twice on the same development prompt before publishing the completion receipt. Repeating a completed cell verifies its archive. Failed attempts stay available for diagnosis. Keep the model snapshots and views at their recorded absolute paths. The original SVD runs on CPU, produces dense adapter matrices and needs substantial RAM, disk and time, particularly for Mamay 27B.

## Evaluate and read redistribution

Use the maintained [Fair-UK evaluator](https://github.com/reirei-00/fair-uk/tree/db435f59ec65890969358af15084737e2e64bf77) on identical held-out examples before and after each intervention. Run this command from the model environment. The SVD adapter is merged into a separate evaluation copy and its development logits are checked after reload. Both evaluation checkpoints use the original model tokenizer with its full context length.

```bash
"$ETHOS_PYTHON" ethos/suite.py evaluate "${common[@]}" "${cell[@]}" \
  --fair-uk "$WORK/fair-uk" --eval-python "$EVAL_PYTHON" \
  --task stereoset_uk --input "$WORK/uk_eval.csv" --bootstrap 0
```

For English, install Fair-UK's `matched-en-v1` release and pass the installed `manifest.json` to the pinned evaluator with `--dataset-bundle`. It contains 949 held-out StereoSet rows, 1,580 WinoBias Controlled rows, 558 source-matched WinoBias Natural rows and 58,492 BBQ rows. The English StereoSet panel uses the original English sentences and shared Ukrainian target labels for target-macro scoring. Ukrainian panels use the files pinned in Fair-UK's native registry. Use `--bootstrap 0` for both StereoSet panels on the seed-42 split because some protected families span Fair-UK strata. WinoBias and BBQ use 1,000 paired bootstrap draws. [Fair-UK release and installation](https://github.com/reirei-00/fair-uk/blob/d888b451b954c7329b6c869bdfe1112268766e64/docs/fair-uk.md#bilingual-data)

To evaluate all eight interventions for one model, list the available panels in a JSON file. Each key is a task ID. Each value supplies `input` and/or `dataset-bundle` paths relative to that file. For example, `{"stereoset_uk":{"input":"uk_eval.csv"}}` evaluates this registered release across four domains and both training languages. Include all eight configured panels to collect 64 pairs per model.

```bash
"$ETHOS_PYTHON" ethos/panel_sweep.py "${common[@]}" --model "$ETHOS_MODEL" \
  --fair-uk "$WORK/fair-uk" --eval-python "$EVAL_PYTHON" --panels "$WORK/panels.json"
```

Add `--execute` to run the printed commands. Select each of the six model environments in turn. Restarting the sweep verifies and reuses completed pairs with the same checkpoint, release, evaluator, runtime and inference settings. Interrupted attempts retain their logs.

Each pair preserves the per-item predictions, Fair-UK reports, paired changes, intervals and coverage. Each StereoSet pair writes `domains.csv` with four rows of target-macro SS, LMS and ICAT. The trained domain identifies a row of the 4 × 4 redistribution matrix and the evaluated domain identifies a column. Read SS against 50, inspect direction changes, and use LMS and ICAT to check whether language-modeling quality also changed. Assess WinoBias and BBQ using their native category and condition breakdowns. A calibrated redistribution index and the other study metrics requiring extra panels or judgments remain separate analyses.

To export one complete matrix, pass the four paired output directories printed by the runner for the same model, training language, evaluation language and seed. The exporter verifies their receipts, base scores, source rows and inference settings. The long-format CSV contains sixteen targeted and off-target comparisons with their native scores, signed changes, direction relative to 50 and parity crossings.

```bash
python3 ethos/domain_matrix.py --pairs "$GENDER_PAIR" "$PROFESSION_PAIR" "$RACE_PAIR" "$RELIGION_PAIR" \
  --output "$WORK/matrix.csv"
```

## Reproduce the software check

With the training stack installed, run the following on CPU. The test creates a disposable model and dataset, runs both upstream training branches and SVD, repeats the training command, exports the checkpoint, runs Fair-UK inference on both sides, and verifies all four domain rows. The optional `--matrix` flag trains all four tiny English interventions and verifies the combined sixteen-row matrix. Use a new output directory for each invocation.

```bash
"$ETHOS_PYTHON" tests/smoke.py --source "$WORK/Ethos" --fair-uk "$WORK/fair-uk" \
  --eval-python "$EVAL_PYTHON" --output "$WORK/smoke-1" --matrix
```

## Reproduce the seed-42 analysis

The [executed notebook](analysis/ethos_redistribution.ipynb) reads the saved Fair-UK scores for all 384 paired evaluations. It shows before/after values for each of the six models in all four combinations of training and evaluation language. Two heatmaps show domain-specific changes in stereotype-score distance from parity and language-modeling score. Separate tables cover WinoBias and BBQ.

The notebook opens with an [18-metric analyst checklist](analysis/metric_checklist.csv) keyed to the rows of the FairForget metric sheet. It reports descriptive before/after results for forget efficacy, LMS, ICAT, M3d language transfer and native BBQ bias by category. The checklist identifies the inputs available for RI and the missing evidence for the other metrics. In the target CSV, positive `forget_efficacy` is improvement. In the domain CSV, positive `bias_change` is an increase in distance from parity.

From this study directory on Linux or macOS, run the following with Python 3.13. After installing the listed packages, replay works offline on CPU. The notebook writes full-precision CSV tables and PNG figures to `analysis/outputs/`. Open the notebook directly to read its saved outputs. Set `ETHOS_ARTIFACT_ROOT` to the study directory if starting the kernel outside the checkout.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r analysis/requirements.txt
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.record_timing=False analysis/ethos_redistribution.ipynb
```

The [score extract](analysis/seed-42/fair-uk-reports.jsonl.gz) contains one record per paired report, with native scores, uncertainty, sample sizes, the evaluator revision and receipt hashes. The notebook checks the uncompressed SHA-256 and reconciles each record with the [verified experiment ledger](jupiter/2026-11-21/status/experiments.csv). An available score may lack a bootstrap interval. Unavailable BBQ changes retain their native status and are excluded from direction counts.

To verify the extract against the original JUPITER files, run the commands below from a checkout of this study with access to the run directory. The collector verifies each paired receipt and its files, the training and export identities, complete predictions and panel coverage. Compare the uncompressed JSON Lines digest with the [published checksum](analysis/seed-42/reports.sha256).

```sh
python3 ethos/collect_paired_reports.py /e/project1/e-dev-2026d09-075/fairforget/runs/ethos-bilingual-5e4a6bf-seed42 /tmp/ethos-fair-uk-reports.jsonl.gz
gzip -dc /tmp/ethos-fair-uk-reports.jsonl.gz | sha256sum
```
