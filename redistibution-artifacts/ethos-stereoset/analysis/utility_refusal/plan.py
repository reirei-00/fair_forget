import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def reference(run, phase):
    attempt = run / read(run / "complete.json")["attempt"]
    (pointer,) = (attempt / "evaluation/stereoset_en").glob("*/complete.json")
    ref = read(pointer)
    pair = pointer.parent / ref["attempt"]
    receipt = pair / "complete.json"
    filename = f"{phase}/run.json"
    if (
        sha(receipt) != ref["receipt_sha256"]
        or sha(pair / filename) != read(receipt)["files"][filename]
    ):
        raise ValueError(f"Invalid reference receipt in {pair}")
    provenance = read(pair / filename)
    identity = provenance["model_identity"]
    export = Path(identity["parameters"]["pretrained"])
    files = identity["local_files_sha256"]
    if not files or not all((export / name).is_file() for name in files):
        raise FileNotFoundError(f"Incomplete export {export}")
    return dict(
        export=str(export),
        files=files,
        tokenizer=provenance["evaluation_asset_identity"]["tokenizer_sha256"],
        run=str(pair / filename),
        sha256=sha(pair / filename),
    )


def build(root, ledger):
    with ledger.open() as source:
        cells = list(csv.DictReader(source))
    models = sorted({cell["model"] for cell in cells})
    keys = {(cell["model"], cell["language"], cell["domain"]) for cell in cells}
    domains = {"gender", "profession", "race", "religion"}
    if (
        len(models) != 6
        or len(cells) != 48
        or keys != set(itertools.product(models, ("en", "uk"), domains))
    ):
        raise ValueError("Incomplete or duplicate study cells")
    bases = {}
    checkpoints = []
    for cell in sorted(cells, key=lambda row: (row["model"], row["language"], row["domain"])):
        model = cell["model"]
        run = root / "results" / cell["run_id"]
        before = reference(run, "before")
        if model in bases:
            previous = bases[model]["reference"]
            if (before["files"], before["tokenizer"]) != (previous["files"], previous["tokenizer"]):
                raise ValueError(f"Baseline differs across checkpoints for {model}")
        else:
            bases[model] = dict(id=model + "__base", model=model, phase="before", reference=before)
        checkpoints.append(
            dict(
                id="__".join((model, cell["language"], cell["domain"])),
                model=model,
                phase="after",
                training_language=cell["language"],
                domain=cell["domain"],
                reference=reference(run, "after"),
            )
        )
    return [
        {**item, "language": lang}
        for item in [*bases.values(), *checkpoints]
        for lang in ("en", "uk")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "ledger", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write(args.output, build(args.root, args.ledger))
