#Stage 1: translation comparison

We select 200 representative StereoSet development items, translates each complete item
with DeepL, Gemini, and Lapa, calculates reference-free scores, and exports one comparison CSV.

## Local

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
jupyter lab
```

Add the keys to `.env`, open `notebooks/stereoset_translation_comparison_200.ipynb`, and run it.
The notebook saves each API result and reuses completed translations on reruns.

## Colab

Upload the notebook, add the same values through Colab Secrets, and run.
