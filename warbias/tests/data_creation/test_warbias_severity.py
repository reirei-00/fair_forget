"""Offline tests for first-pass JEV severity labels (fixture client, no requests)."""

import csv
import json

import pytest
from data_creation_support import JEV_CONFIG as CONFIG
from data_creation_support import JevResponse as Response

from fairforget.warbias import severity
from fairforget.warbias.jev_cases import cases_from


class Client:
    def __init__(self, choice="high", model="jev-2026-09"):
        self.calls, self.choice, self.model = [], choice, model

    def post(self, url, **kwargs):
        self.calls.append(kwargs["json"])
        probabilities = {"high": 0.1, "standard": 0.1, "unscorable": 0.0}
        probabilities[self.choice] = 0.9 - (0.1 if self.choice == "unscorable" else 0)
        probabilities = {k: v for k, v in probabilities.items()}
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}
        return Response(
            {
                "model": self.model,
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {
                    "severity": {
                        "type": "choice",
                        "choice": self.choice,
                        "confidence": 0.8,
                        "probabilities": probabilities,
                    }
                },
            }
        )


def test_preview_sends_nothing_and_execute_saves_raw_and_review_sheet(
    inputs, tmp_path, monkeypatch
):
    cases = cases_from(*inputs)
    assert cases[0]["state"]["claim_uk"].startswith("внутрішньо переміщені")
    assert cases[0]["state"]["example_stereotypes_en"] == ["IDPs collaborate with the enemy."]
    plan = severity.run(cases, CONFIG, tmp_path / "preview")
    assert plan["execute"] is False and not (tmp_path / "preview" / "responses.jsonl").exists()

    monkeypatch.setenv("JEV_API_KEY", "fixture")
    client = Client()
    summary = severity.run(cases, CONFIG, tmp_path / "run", client=client, execute=True)
    assert summary["labels"]["high"] == 1 and len(client.calls) == 1
    assert set(client.calls[0]) == {"model", "state", "questions"}
    label = json.loads((tmp_path / "run" / "labels.jsonl").read_text().splitlines()[0])
    assert label["status"] == "machine_provisional_pending_two_human_raters"
    rows = list(csv.DictReader((tmp_path / "run" / "human_review.csv").open(encoding="utf-8")))
    assert rows[0]["jev_label"] == "high" and rows[0]["rater_a_label"] == ""

    again = Client()
    severity.run(cases, CONFIG, tmp_path / "run", client=again, execute=True)
    assert again.calls == []  # resumed from saved raw responses


def test_invalid_typed_response_is_rejected(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    cases = cases_from(*inputs)
    client = Client()
    original = client.post

    def broken(url, **kwargs):
        response = original(url, **kwargs)
        response.payload["answers"]["severity"]["choice"] = "extreme"
        return response

    client.post = broken
    with pytest.raises(ValueError):
        severity.run(cases, CONFIG, tmp_path / "bad", client=client, execute=True)
    assert not (tmp_path / "bad" / "responses.jsonl").read_text()


class ScoreClient:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(kwargs["json"])
        probabilities = {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}
        return Response(
            {
                "model": "jev-2026-09",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {
                    "severity": {
                        "type": "score",
                        "score": 2.7,
                        "confidence": 0.7,
                        "legend": {str(i): v for i, v in enumerate(severity.SCALE5_LEVELS)},
                        "probabilities": probabilities,
                    }
                },
            }
        )


def test_scale5_maps_levels_to_slider_and_withholds_mechanisms(inputs, tmp_path, monkeypatch):
    cases = cases_from(*inputs)
    question = severity.questions("scale5")["severity"]
    assert question["type"] == "score" and len(question["criteria"]) == 5
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    client = ScoreClient()
    summary = severity.run(
        cases, CONFIG, tmp_path / "s5", client=client, execute=True, rubric="scale5"
    )
    assert "harm_mechanisms_uk" not in client.calls[0]["state"]
    label = json.loads((tmp_path / "s5" / "labels.jsonl").read_text())
    assert label["jev_label"] == "4"
    assert label["jev_score"] == pytest.approx(3.7)
    assert label["jev_probabilities"]["5"] == 0.1
    assert summary["labels"]["4"] == 1
    assert summary["mean_expected_score"] == pytest.approx(3.7)
    with (tmp_path / "s5" / "human_review.csv").open(encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["jev_score"] == "3.7"


@pytest.mark.parametrize("change", ["evidence", "rubric", "model"])
@pytest.mark.parametrize("execute", [False, True])
def test_resume_rejects_changed_requests_before_writing(
    inputs, tmp_path, monkeypatch, change, execute
):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    cases = cases_from(*inputs)
    output = tmp_path / "resume"
    severity.run(cases, CONFIG, output, client=Client(), execute=True)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    config = dict(CONFIG)
    rubric = "v1"
    if change == "evidence":
        cases[0]["state"]["claim_uk"] = "A different claim"
    elif change == "rubric":
        rubric = "v3"
    else:
        config["model"] = "different-judge"
    client = Client()
    with pytest.raises(ValueError, match="saved JEV request differs"):
        severity.run(cases, config, output, client=client, execute=execute, rubric=rubric)
    assert client.calls == []
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_resume_preserves_human_review_edits(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "fixture")
    cases = cases_from(*inputs)
    output = tmp_path / "review"
    severity.run(cases, CONFIG, output, client=Client(), execute=True)
    path = output / "human_review.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0].update(
        rater_a_label="standard",
        rater_b_label="high",
        agreed_label="high",
        comment="Reviewed independently",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    client = Client()
    severity.run(cases, CONFIG, output, client=client, execute=True)
    assert client.calls == []
    assert path.read_bytes() == before
