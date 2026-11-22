import json
import struct
import tempfile
import unittest
from pathlib import Path

from collect import paired
from plan import sha, write
from text_baseline import convert_shard, header


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

    def test_utility_requires_identical_questions_and_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                    inputs_sha256="inputs",
                    language="uk",
                    versions={},
                    worker="code",
                    settings={"pretrained": phase},
                    reference=phase,
                )
                write(out / "request.json", binding)
                file = out / "belebele.jsonl"
                row = dict(
                    id=1,
                    task="test",
                    doc_hash="item",
                    prompt_hash="prompt",
                    target=1,
                    correct=correct,
                )
                file.write_text(json.dumps(row) + "\n")
                receipt = {
                    "results": {"belebele": {"rows": 1, "sha256": sha(file)}},
                    "request_sha256": sha(out / "request.json"),
                }
                write(out / "complete.json", receipt)
                items.append(item)
            self.assertEqual(paired(root, *items, "belebele")["change_pp"], 100)
            row["prompt_hash"] = "different"
            file.write_text(json.dumps(row) + "\n")
            receipt["results"]["belebele"]["sha256"] = sha(file)
            write(out / "complete.json", receipt)
            with self.assertRaisesRegex(ValueError, "different utility"):
                paired(root, *items, "belebele")


if __name__ == "__main__":
    unittest.main()
