# WarBias dataset construction

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
records all 21 added dimensions and their original and retained counts.
[The evidence mapping](protocol/inputs/cohort_evidence.json) records supporting
references and pending source checks, including their case or profile scope.

The separate BBQ-style QA set contains **240 questions per language** from
40 cases: 120 veteran and 120 IDP questions. Each scenario has six variants.
[Ukrainian QA](data/qa_uk.jsonl), [English QA](data/qa_en.jsonl) and their
[protocol](protocol/qa.json) include answer keys and review status. The 480
Ukrainian human reviews remain bound to the exact text each person saw.
Assistant resolutions are marked separately. GPT-6 Astra translated all English
QA text; it has no human translation review.

The recorded design selected 1,000 base inputs and 200 profile extensions.
Excluding 136 occupation profiles and five older women veteran profiles left
1,059 items for human annotation. Excluding 14 legitimacy-category items and
39 items in six local cases leaves the selected 1,006. One author-edited feedback
case adds one triplet, giving the current 1,007.
The taxonomy marks original cohort membership with
`in_original_annotation_cohort`. The [cohort protocol](protocol/cohort.json)
defines the original selected inputs; the frozen allocation and pilot ratings
retain their original scope.
The [construction guide](docs/warbias-construction.md)
explains the selector, extensions and exclusions.

Keep the intentional `the-project/` namespace in this artifact’s repository
references.

The package contains dataset text, construction inputs and reproducible methods.
Per-item provenance separates original annotated text from author revisions.
Triplet-quality ratings apply only to unchanged text. The taxonomy and triplets
store `severity_mean_score` (0–100) and `severity_level` (1–5). Notebook 6 derives
both from case-level human ratings; insufficient compatible ratings leave nulls. A new model run reproduces the
selected inputs; its sentences can differ from saved dataset text.

## Install

Use Python 3.10+ and a separate environment. Run from `warbias-artifacts/` or an
exported package:

```sh
python -m pip install -e '.[dev,notebooks]'
warbias_kit doctor
```

The package is `warbias_kit-warbias`; its Python namespace and command are
`warbias_kit`. Add `generation` for local vLLM inference or `quality` for COMET.
The package includes its data-release validator.

## Walkthroughs

| Notebook | Purpose |
| --- | --- |
| [1. Ukrainian pilot](notebooks/01_ukrainian_pilot.ipynb) | Saved Mamay/Lapa samples and two prompt methods |
| [2. Model selection](notebooks/02_model_selection.ipynb) | Human pilot scores and the recorded Lapa/sequential choice |
| [3. Ukrainian generation](notebooks/03_full_ukrainian_generation.ipynb) | Reproduce the original allocation and inspect the current 1,007 items |
| [4. Sample translation and COMET](notebooks/04_sample_translation_comet.ipynb) | Recorded Ukrainian-to-English translations and quality scores |
| [5. English translation](notebooks/05_full_english_translation.ipynb) | Verify all 1,007 Astra translations and their source hashes |
| [6. Case severity](notebooks/06_severity.ipynb) | Calculate case means and severity levels for both languages |
| [7. QA dataset](notebooks/07_qa_dataset.ipynb) | Build six variants per scenario, check human annotations and inspect the bilingual release |

Run the notebooks with a local Python kernel:

```sh
python scripts/run_notebooks.py
```

Default execution reads saved evidence and prepares model requests. Explicit
run flags enable generation or COMET inference. Local outputs stay
in ignored `notebooks/runs/`. Set model versions, providers and settings in
`configs/` before executing a model.

## Verify and export

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
