# FairForget

Research workflows for benchmark adaptation, translation and unlearning.

## Translation comparison

The notebooks load a benchmark sample, translate complete items with DeepL, Gemini, and Lapa,
validate the translations, calculate COMET, MetricX, and LaBSE scores, and export review files.

## Notebooks

- `notebooks/not_executed/` contains clean notebooks for a new run.
- `notebooks/executed/` contains completed notebooks with saved outputs.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
jupyter lab
```

Add `DEEPL_AUTH_KEY`, `GEMINI_API_KEY`, and `LAPA_API_KEY` to `.env`. Open a notebook from
`notebooks/not_executed/` and run its cells from top to bottom. Completed translations are saved
after each batch and reused when the notebook is run again.

## Run in Colab

Upload a clean notebook, add the same three keys through Colab Secrets, and run its cells from top
to bottom.

## WarBias subproject

The WarBias pipelines live in [`warbias/`](warbias/): the Ukrainian generation
pilot, blinded human model/strategy selection, full generation, English
translation and review. Both aligned language versions are retained.
See the [WarBias guide](warbias/README.md) for its separate installation, tests
and source export. Existing benchmark notebooks, evaluation scripts and
their root dependencies remain independent of this subproject.

## Unlearning

The [unlearning notebooks](notebooks/README.md) contain one notebook per method:
WGA+KL, NPO+KL and BiasUnlearn-style. All three use the same final WarBias
sources and component split. Each trains IDP and veteran groups separately,
in separate Ukrainian and English directions. All six pinned models start from
their base checkpoints in every direction: 24 runs per method, 72 in total.
Evaluation covers both groups in Ukrainian and English. Start with the
[setup notebook](notebooks/setup.ipynb) or [setup guide](notebooks/SETUP.md).
The [configuration](configs/unlearning/warbias_final.json) defines the methods
and shared training settings.

The [WarBias workflow](notebooks/WARBIAS_UNLEARNING.md) defines data roles and
evaluation requirements. All metrics use Fair-UK. RMU and task-vector notebooks
remain separate experiments.

Saved unlearning reports remain under [executed notebooks](notebooks/executed/).
