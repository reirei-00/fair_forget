"""Dataset export order stays separate from canonical evidence hashing."""

import json

import pytest

from warbias_kit.io import digest, read_rows, write_rows


def test_explicit_columns_preserve_values_and_evidence_identity(tmp_path):
    rows = [{"metadata": "дані", "text": "Текст", "id": "one"}]
    path = tmp_path / "dataset.jsonl"
    write_rows(path, rows, columns=["id", "text", "metadata"])
    written = read_rows(path)
    assert list(written[0]) == ["id", "text", "metadata"]
    assert written == rows
    assert digest(written) == digest(rows)
    write_rows(path, rows)
    assert list(json.loads(path.read_text())) == ["id", "metadata", "text"]


@pytest.mark.parametrize("columns", ["id", [], ["id", "id"], [""], [None], [["id"]]])
def test_rejects_invalid_column_declarations(tmp_path, columns):
    with pytest.raises(ValueError, match="Columns"):
        write_rows(tmp_path / "rows.jsonl", [{"id": "one"}], columns=columns)


@pytest.mark.parametrize("row", [{"id": "one", "extra": True}, {}, None])
def test_rejects_rows_that_do_not_match_declared_columns(tmp_path, row):
    with pytest.raises(ValueError, match="fields must match"):
        write_rows(tmp_path / "rows.jsonl", [row], columns=["id"])
