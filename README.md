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

The [BiasUnlearn](warbias-artifacts/notebooks/10_biasunlearn.ipynb) and [WGA+KL](warbias-artifacts/notebooks/11_wga_kl.ipynb)
notebooks use the final WarBias sources and component split. Each method trains
six models for one epoch, separately for IDPs and veterans in Ukrainian and
English: 48 runs in total. Each adapter is evaluated only on its training group
and language. Start with the [setup notebook](warbias-artifacts/notebooks/09_unlearning_setup.ipynb) or
[setup guide](warbias-artifacts/docs/unlearning-setup.md). The [configuration](warbias-artifacts/configs/unlearning/warbias_final.json)
defines the methods and training settings.

The [WarBias workflow](warbias-artifacts/docs/unlearning.md) defines data roles and
evaluation requirements. All metrics use Fair-UK. MMLU and Belebele were not run
for these adapters. Other methods and saved experiments remain separate.

Saved unlearning reports remain under [executed notebooks](notebooks/executed/).
