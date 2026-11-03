# WarBias dataset

WarBias is a Ukrainian–English benchmark of stereotypes about veterans and internally displaced people: 1,007 sentence triplets and 240 multiple-choice questions per language, built from a taxonomy of 120 claims in 27 categories.

WarBias uses blinded human ratings of Ukrainian samples to inform model and
prompt selection. It then generates Ukrainian triplets with Lapa and sequential
prompts and translates them into English. Both languages keep matching item IDs.

The dataset contains **1,007 aligned triplets per language in 120 semantic cases
and 27 categories: 598 veterans and 409 internally displaced people (IDPs)**.
The [Ukrainian](data/uk.jsonl) and [English](data/en.jsonl) files share IDs and
metadata. The [dataset manifest](protocol/dataset.json) binds all release files.
English triplet translations use GPT-6 Astra; they have not undergone human translation
review. Dataset text is licensed under [CC BY-SA 4.0](data/LICENSE.md).
[The bilingual taxonomy](protocol/inputs/cohort_taxonomy.csv) lists every case,
category, claim and source. It includes only cases with `case_scope = general`.
[The attribute catalog](protocol/inputs/cohort_attributes.json)
records all 21 attribute dimensions, their values and selection counts.
[The evidence mapping](protocol/inputs/cohort_evidence.json) records supporting
references and pending source checks, including their case or profile scope.

The separate BBQ-style QA set contains **240 questions per language** from
40 cases: 120 veteran and 120 IDP questions. Each scenario has six variants.
[Ukrainian QA](data/qa_uk.jsonl), [English QA](data/qa_en.jsonl) and their
[protocol](protocol/qa.json) include answer keys and review status. The 480
Ukrainian human reviews remain bound to the exact text each person saw.
The Ukrainian items include 185 with independent human acceptance and 55 with
[author approval](protocol/inputs/qa_author_reviews.jsonl): 28 revised and 27
retained items. Author decisions are separate from independent annotation.
GPT-6 Astra translated all English QA text; it has no human translation review.

Both languages use the same column order for each dataset type: ID and text
first, followed by answers and metadata. The dataset manifests declare the order
used in JSONL and Parquet.

The [construction guide](docs/warbias-construction.md) defines the taxonomy
allocation, profile extensions, exclusions and feedback case that produce the
1,007 generation inputs. The [cohort protocol](protocol/cohort.json) lists their
IDs. `in_original_annotation_cohort` distinguishes cohort cases from the feedback case.
The taxonomy's `triplet_count` counts generation inputs; the dataset manifest's
`counts.triplets_by_case` counts released items after
[metadata corrections](protocol/dataset_metadata_corrections.json).

The package contains dataset text, construction inputs and reproducible methods.
Per-item provenance identifies reviewed text and author-edited text.
Triplet-quality ratings apply only to unchanged text. The taxonomy and triplets
store `severity_mean_score` (0–100) and `severity_level` (1–5). Notebook 6 derives
both from two active human ratings per case, including ratings with an unrecorded
rubric version. All 120 cases and 1,007 triplets per language have severity values.
Generation uses fixed inputs; model outputs can differ from the released text.

## Install

Use Python 3.10+ and a separate environment. Run from `warbias-artifacts/` or an
exported package:

```sh
python -m pip install -e '.[dev,notebooks]'
python -m pip install -e ./evaluator
warbias_kit doctor
```

The package is `warbias_kit-warbias`; its Python namespace and command are
`warbias_kit`. Add `generation` for local vLLM inference, `quality` for COMET, or install
`requirements-unlearning.txt` for adapter training.
The package includes its data-release validator. The `evaluator/` directory
contains the local evaluation library; see the [evaluation guide](docs/evaluation.md).

## Walkthroughs

| Notebook | Purpose |
| --- | --- |
| [1. Ukrainian pilot](notebooks/01_ukrainian_pilot.ipynb) | Saved Mamay/Lapa samples and two prompt methods |
| [2. Model selection](notebooks/02_model_selection.ipynb) | Human pilot scores and the recorded Lapa/sequential choice |
| [3. Ukrainian generation](notebooks/03_full_ukrainian_generation.ipynb) | Prepare 1,007 inputs and generate S, C and U sequentially |
| [4. Sample translation and COMET](notebooks/04_sample_translation_comet.ipynb) | Recorded Ukrainian-to-English translations and quality scores |
| [5. English translation](notebooks/05_full_english_translation.ipynb) | Verify all 1,007 Astra translations and their source hashes |
| [6. Case severity](notebooks/06_severity.ipynb) | Calculate case means and severity levels for both languages |
| [7. QA sampling and reproduction](notebooks/07_qa_dataset.ipynb) | Rebuild all QA items, sample complete scenarios and export aligned language files |
| [8. Baseline evaluation](notebooks/08_warbias_baseline.ipynb) | Six-model triplet and QA results with severity slices; [run guide](docs/evaluation.md) |
| [9. Unlearning setup](notebooks/09_unlearning_setup.ipynb) | Prepare model assets and component-level data splits |
| [10. BiasUnlearn](notebooks/10_biasunlearn.ipynb) | Train separate group and language adapters for one epoch |
| [11. WGA+KL](notebooks/11_wga_kl.ipynb) | Train stereotype-suppression adapters with a KL penalty |
| [12. After-intervention evaluation](notebooks/12_after_intervention.ipynb) | Compare each adapter with its base on the same group and language |
| [13. Native benchmark evaluation](notebooks/13_native_evaluation.ipynb) | Prepare optional capability and native benchmark comparisons |

The [unlearning protocol](docs/unlearning.md) defines the two methods, six
models and four group/language directions. The [setup guide](docs/unlearning-setup.md)
describes model downloads and training. Evaluation uses the bundled library.

Run all thirteen notebooks with a local Python kernel:

```sh
python scripts/run_notebooks.py
```

Default execution reads saved evidence and prepares model requests. Explicit
run flags enable generation, COMET inference, training or evaluation. Local outputs stay
in ignored `notebooks/runs/`. Set model versions, providers and settings in
`configs/` before executing a model.

## Verify and export

Install `requirements-unlearning.txt` to include the training tests.

```sh
python -m pytest tests -q
python -m ruff check src tests scripts notebooks
python scripts/export_dataset.py build --output /path/to/new-warbias-package
python scripts/export_dataset.py verify --directory /path/to/new-warbias-package
```

The source export contains both language files, settings, prompts, study inputs,
notebooks and tests.
Keep credentials, model weights, raw annotation exports and service responses in
local storage. Prompt hashes are in `prompts/manifest.json`.
Use the exported files in a fresh anonymous repository. The source repository,
pull requests and Git history are not part of the anonymous export.
