"""Run the dataset notebooks in order and retain their actual outputs."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from export_dataset import source_root


def run(source, *, names=None, run_directory=None, timeout=300):
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    project = source_root(source)
    directory = project / "notebooks"
    paths = sorted(directory.glob("[0-9][0-9]_*.ipynb"))
    selected = set(names or (path.stem for path in paths))
    if not paths or selected - {path.stem for path in paths}:
        raise ValueError("Select existing numbered dataset notebooks")
    run_directory = run_directory or os.environ.get("WARBIAS_NOTEBOOK_RUN")
    if run_directory is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_directory = f"runs/walkthrough/{stamp}-{uuid4().hex[:8]}"
    relative = Path(run_directory)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The run directory must be relative to notebooks/")
    output = directory / relative
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "WARBIAS_NOTEBOOK_RUN": relative.as_posix()}
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(project / "src"), environment.get("PYTHONPATH")])
    )
    completed = []
    records = []
    with tempfile.TemporaryDirectory(prefix="warbias-kernel-") as temporary:
        kernel = Path(temporary) / "warbias-python"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Python 3",
                    "language": "python",
                }
            )
        )
        for path in paths:
            if path.stem not in selected:
                continue
            notebook = nbformat.read(path, as_version=4)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            manager = KernelManager(
                kernel_name="warbias-python",
                kernel_spec_manager=KernelSpecManager(
                    kernel_dirs=[temporary], ensure_native_kernel=False
                ),
            )
            client = NotebookClient(
                notebook,
                km=manager,
                timeout=timeout,
                record_timing=False,
                resources={"metadata": {"path": str(directory)}},
            )
            started = time.monotonic()
            try:
                client.execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            completed.append((path, notebook))
            records.append(
                {
                    "notebook": path.name,
                    "source_notebook_sha256": before,
                    "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
    for (path, notebook), record in zip(completed, records):
        nbformat.write(notebook, path)
        record["executed_notebook_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"status": "complete", "run_directory": relative.as_posix(), "notebooks": records}
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--notebook", action="append", dest="names", metavar="STEM")
    parser.add_argument("--run-directory")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = run(
        args.source, names=args.names, run_directory=args.run_directory, timeout=args.timeout
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
