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

## JUPITER unlearning results

The [executed October RMU notebook](notebooks/executed/jupiter_rmu_october_2026.ipynb)
replays the completed 23–24 October 2026 Lapa, Gemma and Qwen experiments from
saved English BBQ predictions, with before/after tables and training-loss plots.
The [clean notebook](notebooks/not_executed/jupiter_rmu_october_2026.ipynb) can be
run locally with `pandas`, `matplotlib` and Jupyter; it needs no GPU or API keys.
The [evidence package](evaluation/unlearning/jupiter_rmu_20261024/README.md)
contains original metrics, compressed predictions, configurations, job receipts,
the preserved historical scorer and a source snapshot. This is a separate
English pilot, rather than a completed current Fair-UK intervention panel.

The [JUPITER training workflow](notebooks/not_executed/jupiter_rmu_training_workflow.ipynb)
loads the preserved trainer, checks local weights, input files and CUDA, and runs
adaptive RMU plus full BBQ evaluation into a new timestamped directory. It was
prepared on 30 October and has not been executed for a new training run.
The historical replay also reports the incomplete target mapping: 5,656 of
5,672 BBQ items; 16 unmapped items remain in the original scoring denominators.
