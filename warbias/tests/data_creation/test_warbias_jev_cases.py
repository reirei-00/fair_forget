"""Shared resume and review-sheet guarantees for both JEV annotation workflows."""

import copy
import csv
import json

import pytest
from data_creation_support import JEV_CONFIG, JevResponse

from fairforget.warbias import harm_mechanisms, severity
from fairforget.warbias.jev_cases import cases_from


class Client:
    def __init__(self, model="fixture-judge"):
        self.model, self.calls = model, []

    def post(self, _url, **kwargs):
        self.calls.append(kwargs["json"])
        answers = {}
        for name, question in kwargs["json"]["questions"].items():
            choices = list(question["criteria"])
            answers[name] = {
                "type": "choice",
                "choice": choices[0],
                "confidence": 1.0,
                "probabilities": {key: float(key == choices[0]) for key in choices},
            }
        return JevResponse(
            {
                "model": self.model,
                "usage": {"input_tokens": 5, "output_tokens": 1},
                "answers": answers,
            }
        )


@pytest.fixture(params=[severity, harm_mechanisms], ids=["severity", "mechanisms"])
def workflow(request, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    return request.param


def snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


def test_wrong_judge_is_rejected_before_saving(workflow, inputs, tmp_path):
    cases = cases_from(*inputs)
    output = tmp_path / "new"
    config = {**JEV_CONFIG, "expected_returned_model": "required-judge"}
    with pytest.raises(ValueError, match="frozen expectation"):
        workflow.run(cases, config, output, client=Client(), execute=True)
    assert not (output / "responses.jsonl").read_text()
    assert not (output / "labels.jsonl").exists()


@pytest.mark.parametrize("execute", [False, True])
def test_saved_judge_is_checked_before_writing(workflow, inputs, tmp_path, execute):
    cases = cases_from(*inputs)
    output = tmp_path / "saved"
    workflow.run(cases, JEV_CONFIG, output, client=Client(), execute=True)
    before = snapshot(output)
    config = {**JEV_CONFIG, "expected_returned_model": "required-judge"}
    client = Client()
    with pytest.raises(ValueError, match="frozen expectation"):
        workflow.run(cases, config, output, client=client, execute=execute)
    assert client.calls == []
    assert snapshot(output) == before


def test_append_cases_preserves_review_bytes_and_adds_coverage(workflow, inputs, tmp_path):
    cases = cases_from(*inputs)
    output = tmp_path / "extended"
    config = {**JEV_CONFIG, "expected_returned_model": "fixture-judge"}
    workflow.run(cases, config, output, client=Client(), execute=True)
    sheet = output / "human_review.csv"
    with sheet.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["comment"] = "Human review: keep this, including punctuation."
    with sheet.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = sheet.read_bytes()
    added = copy.deepcopy(cases[0])
    added["case_id"] = "WB-NEW"
    added["state"]["claim_uk"] = "Another claim"
    client = Client()
    summary = workflow.run(cases + [added], config, output, client=client, execute=True)
    assert len(client.calls) == 1 and summary["cases"] == 2
    assert sheet.read_bytes().startswith(before)
    with sheet.open(encoding="utf-8", newline="") as stream:
        reviewed = list(csv.DictReader(stream))
    assert [row["case_id"] for row in reviewed] == [cases[0]["case_id"], "WB-NEW"]
    assert reviewed[0]["comment"] == rows[0]["comment"]
    assert reviewed[1]["comment"] == ""
    final = snapshot(output)
    client = Client()
    workflow.run(cases + [added], config, output, client=client, execute=True)
    assert client.calls == [] and snapshot(output) == final
    with pytest.raises(ValueError, match="absent from the current selection"):
        workflow.run(cases, config, output, client=client, execute=True)
    assert snapshot(output) == final


def test_duplicate_saved_response_is_not_silently_reused(workflow, inputs, tmp_path):
    cases = cases_from(*inputs)
    output = tmp_path / "duplicates"
    workflow.run(cases, JEV_CONFIG, output, client=Client(), execute=True)
    raw = output / "responses.jsonl"
    record = json.loads(raw.read_text())
    raw.write_text(raw.read_text() + json.dumps(record) + "\n")
    before = snapshot(output)
    client = Client()
    with pytest.raises(ValueError, match="duplicate saved JEV response"):
        workflow.run(cases, JEV_CONFIG, output, client=client, execute=True)
    assert client.calls == [] and snapshot(output) == before
