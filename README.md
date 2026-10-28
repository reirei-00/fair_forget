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
