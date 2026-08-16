# StereoSet translation comparison

The notebook selects 40 representative StereoSet development items, translates each complete item
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

Add the keys to `.env`, open `stereoset_translation_comparison_40.ipynb`, and run it.
The notebook saves each API result and skips completed translations when rerun (to do not overspend because we rely on paid APIs, of you need different behaviour, please edit).

## Colab

Upload the notebook, add the same values through Colab Secrets, and run.
