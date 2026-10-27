"""The quality notebook previews without fetching data or loading a model."""

import json
from pathlib import Path


def test_quality_notebook_preview_is_offline(monkeypatch):
    from fairforget.translation import quality, reference

    def unexpected(*args, **kwargs):
        raise AssertionError("A notebook preview must not fetch data or load COMET")

    monkeypatch.setattr(reference, "load_reference", unexpected)
    monkeypatch.setattr(quality, "load_comet_scorer", unexpected)
    path = Path(__file__).resolve().parents[2] / "notebooks/paper/warbias_translation_quality.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), path.name, "exec"), namespace)
    assert namespace["result"]["status"] == "not_run"
    assert namespace["result"]["input_kind"] == "reference"
