import json
import struct
import tempfile
import unittest
from pathlib import Path

from collect import paired, predictions
from plan import sha, write
from text_baseline import convert_shard, header
from worker_entry import verify_completed


class FollowupTests(unittest.TestCase):
    def test_repacked_text_preserves_binary_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = {
                "vision_tower.v": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
                "language_model.model.layer": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [3, 7],
                },
            }
            metadata = json.dumps(index).encode()
            original = root / "original"
            original.write_bytes(
                struct.pack("<Q", len(metadata)) + metadata + b"abc\x00\x00\x01\xff"
            )
            output = root / "converted"
            hashes, size = convert_shard(original, output, ["language_model.model.layer"])
            self.assertEqual((size, list(hashes)), (4, ["model.layer"]))
            with output.open("rb") as file:
                fields, _ = header(file)
                self.assertEqual(
                    fields["model.layer"], {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
                )
                self.assertEqual(file.read(), b"\x00\x00\x01\xff")

    def test_shutdown_recovery_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = {"id": "model", "language": "uk", "reference": {"sha256": "original"}}
            write(root / "plan.json", [item])
            out = root / "full/model/uk/utility"
            out.mkdir(parents=True)
            write(out / "request.json", {"reference": "original"})
            results = {}
            for name, n in (("belebele", 900), ("mmlu", 14042)):
                path = out / f"{name}.jsonl"
                path.write_text("{}\n" * n)
                results[name] = {"rows": n, "sha256": sha(path)}
            write(
                out / "complete.json",
                {"request_sha256": sha(out / "request.json"), "results": results},
            )
            self.assertEqual(verify_completed(root, 0, "utility"), out)
            (out / "mmlu.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "Incomplete mmlu outputs"):
                verify_completed(root, 0, "utility")

    def test_mmlu_requires_the_registered_subjects_even_for_interim_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subjects = [f"global_mmlu_full_uk_subject_{n}" for n in range(57)]
            write(root / "inputs.json", {f"tasks/mmlu_uk/{s}.yaml": s for s in subjects})
            item = dict(id="base", language="uk", reference={"sha256": "base"})
            out = root / "full/base/uk/utility"
            out.mkdir(parents=True)
            write(
                out / "request.json",
                dict(
                    reference="base",
                    language="uk",
                    track="utility",
                    phase="full",
                    plan_sha256="plan",
                    inputs_sha256=sha(root / "inputs.json"),
                ),
            )
            rows = [
                dict(
                    task=subjects[i % 57], id=i, doc_hash="d", prompt_hash="p", target=1, correct=1
                )
                for i in range(14042)
            ]
            file = out / "mmlu.jsonl"
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            records, _, _ = predictions(root, item, "mmlu", True, "plan")
            self.assertEqual(len(records), 14042)
            rows[0]["task"] = "global_mmlu_full_en_subject_0"
            file.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Prediction tasks"):
                predictions(root, item, "mmlu", True, "plan")

    def test_utility_requires_matching_language_benchmark_and_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "plan.json", ["paired fixture"])
            write(root / "inputs.json", {})
            plan_digest = sha(root / "plan.json")
            items = []
            for phase, correct in (("before", 0), ("after", 1)):
                item = dict(
                    id=phase,
                    model="m",
                    language="uk",
                    training_language="en",
                    domain="gender",
                    reference={"sha256": phase},
                )
                out = root / "full" / phase / "uk" / "utility"
                out.mkdir(parents=True)
                binding = dict(
                    evaluator="code",
                    inputs_sha256=sha(root / "inputs.json"),
                    language="uk",
                    track="utility",
                    phase="full",
                    plan_sha256=plan_digest,
                    versions={},
                    worker="code",
                    settings={"pretrained": phase},
                    reference=phase,
                )
                write(out / "request.json", binding)
                file = out / "belebele.jsonl"
                row = dict(
                    id=1,
                    task="belebele_ukr_Cyrl",
                    doc_hash="item",
                    prompt_hash="prompt",
                    target=1,
                    correct=correct,
                )
                file.write_text("".join(json.dumps({**row, "id": i}) + "\n" for i in range(900)))
                receipt = {
                    "results": {
                        "belebele": {"task": "belebele_ukr_Cyrl", "rows": 900, "sha256": sha(file)}
                    },
                    "request_sha256": sha(out / "request.json"),
                }
                write(out / "complete.json", receipt)
                items.append(item)
            self.assertEqual(paired(root, *items, "belebele")["change_pp"], 100)
            for item in items:
                workdir = root / "full" / item["id"] / "uk/utility"
                request = json.loads((workdir / "request.json").read_text())
                write(workdir / "request.json", dict(request, language="en"))
            with self.assertRaisesRegex(ValueError, "Request language"):
                paired(root, *items, "belebele")
            for item in items:
                workdir = root / "full" / item["id"] / "uk/utility"
                request = json.loads((workdir / "request.json").read_text())
                write(workdir / "request.json", dict(request, language="uk"))
            receipt["results"]["belebele"]["task"] = "belebele_eng_Latn"
            write(out / "complete.json", receipt)
            with self.assertRaisesRegex(ValueError, "Invalid utility predictions"):
                paired(root, *items, "belebele")
            receipt["results"]["belebele"]["task"] = "belebele_ukr_Cyrl"
            file.write_text(
                "".join(
                    json.dumps({**row, "id": i, "task": "belebele_eng_Latn"}) + "\n"
                    for i in range(900)
                )
            )
            receipt["results"]["belebele"]["sha256"] = sha(file)
            write(out / "complete.json", receipt)
            with self.assertRaisesRegex(ValueError, "Prediction tasks"):
                paired(root, *items, "belebele")
            row["prompt_hash"] = "different"
            file.write_text("".join(json.dumps({**row, "id": i}) + "\n" for i in range(900)))
            receipt["results"]["belebele"]["sha256"] = sha(file)
            write(out / "complete.json", receipt)
            with self.assertRaisesRegex(ValueError, "different utility"):
                paired(root, *items, "belebele")


if __name__ == "__main__":
    unittest.main()
