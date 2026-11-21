"""Plan or execute a model's bilingual Fair-UK comparisons from explicit local panel releases."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from suite_data import HERE, SPEC, load_suite, read


def commands(a):
    manifest = load_suite(a.suite)
    spec = read(SPEC)
    if a.model not in spec["models"]:
        raise ValueError("Unknown model")
    sources = read(a.panels)
    allowed = set(sum(spec["panels"].values(), []))
    if not sources or set(sources) - allowed:
        raise ValueError("Select at least one configured evaluation task")
    for task, inputs in sources.items():
        if not inputs or set(inputs) - {"input", "dataset-bundle"}:
            raise ValueError(f"Supply a local release for {task}")
        flags = [
            x
            for k, v in inputs.items()
            for x in ("--" + k, str((a.panels.parent / v).resolve(strict=True)))
        ]
        for cell in manifest["cells"]:
            if cell["model"] != a.model:
                continue
            yield [
                sys.executable,
                str(HERE / "suite.py"),
                "evaluate",
                "--suite",
                str(a.suite.resolve()),
                "--runs",
                str(a.runs.resolve()),
                "--model",
                a.model,
                "--language",
                cell["language"],
                "--domain",
                cell["domain"],
                "--task",
                task,
                "--fair-uk",
                str(a.fair_uk.resolve()),
                "--eval-python",
                str(a.eval_python.absolute()),
                "--bootstrap",
                str(a.bootstrap),
                "--device",
                a.device,
                "--dtype",
                a.dtype,
                *flags,
            ]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("suite", "runs", "panels", "fair-uk", "eval-python"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    for cmd in list(commands(a)):
        print(shlex.join(cmd), flush=True)
        if a.execute:
            subprocess.run(cmd, check=True)
