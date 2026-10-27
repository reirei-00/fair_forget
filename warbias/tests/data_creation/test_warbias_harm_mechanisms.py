"""Offline tests for preliminary JEV harm-mechanism coding."""

import csv
import json

from data_creation_support import JEV_CONFIG as CONFIG
from data_creation_support import JevResponse as Response

from fairforget.warbias import harm_mechanisms as hm
from fairforget.warbias.jev_cases import cases_from


class Client:
    def __init__(self, present):
        self.present, self.calls = set(present), []

    def post(self, url, **kwargs):
        self.calls.append(kwargs["json"])
        answers = {}
        for name in kwargs["json"]["questions"]:
            choice = "present" if name in self.present else "absent"
            probs = {"present": 0.1, "absent": 0.1, "unscorable": 0.0}
            probs[choice] = 0.9
            answers[name] = {
                "type": "choice",
                "choice": choice,
                "confidence": 0.9,
                "probabilities": probs,
            }
        return Response(
            {
                "model": "jev-x",
                "usage": {"input_tokens": 5, "output_tokens": 1},
                "answers": answers,
            }
        )


def test_derive_severity_mapping():
    assert hm.derive_severity(["competence_autonomy", "fraud_faking"])["severity_3"] == "high"
    assert hm.derive_severity(["pathologisation"])["severity_2"] == "standard"
    assert hm.derive_severity(["pathologisation"])["severity_3"] == "medium"
    assert hm.derive_severity([])["severity_3"] is None


def test_run_codes_mechanisms_without_prior_notes(inputs, tmp_path, monkeypatch):

    monkeypatch.setenv("JEV_API_KEY", "fixture")
    cases = cases_from(*inputs)
    client = Client(["disloyalty_collective_blame", "competence_autonomy"])
    summary = hm.run(cases, CONFIG, tmp_path / "out", client=client, execute=True)
    assert "harm_mechanisms_uk" not in client.calls[0]["state"]
    assert set(client.calls[0]["questions"]) == set(hm.CODEBOOK)
    assert summary["severity_3"]["high"] == 1
    label = json.loads((tmp_path / "out" / "labels.jsonl").read_text().splitlines()[0])
    assert label["severity_2"] == "high"
    rows = list(csv.DictReader((tmp_path / "out" / "human_review.csv").open(encoding="utf-8")))
    assert rows[0]["jev_disloyalty_collective_blame"] == "present"
    assert rows[0]["rater_a_fraud_faking"] == ""
