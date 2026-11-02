"""Source, split and loss-role checks for the shared final WarBias streams."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fairforget.unlearning import warbias_data

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class WarBiasDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "source"
        for spec in warbias_data.SOURCES.values():
            destination = self.repo / spec["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / spec["path"], destination)
        self.output = self.root / "streams"

    def test_shared_roles_use_complete_final_ukrainian_sentences(self):
        binding = warbias_data.prepare(self.repo, self.output)
        self.assertEqual(warbias_data.validate_bundle(self.output, self.repo), binding)
        source = {
            row["id"]: row
            for row in read_rows(self.repo / warbias_data.SOURCES["triplets_uk"]["path"])
        }
        self.assertEqual(len(source), 1007)
        ids = {}
        role_ids = set()
        for role, field in (
            ("forget", "stereotype"),
            ("correction", "counter_stereotype"),
            ("retain", "unrelated"),
        ):
            spec = binding["streams"][role]
            rows = read_rows(self.output / spec["path"])
            ids[role] = [row["source_item_id"] for row in rows]
            self.assertEqual(len(rows), spec["rows"])
            for row in rows:
                self.assertNotIn(row["id"], role_ids)
                role_ids.add(row["id"])
                self.assertEqual(row["id"], f"{row['source_item_id']}:{role}")
                self.assertEqual(row["prompt"], "")
                self.assertEqual(row["target"], source[row["source_item_id"]][field])
                self.assertEqual(row["completion_character_span"], [0, len(row["target"])])
                self.assertEqual(row["language"], "uk")
                self.assertEqual(row["source_field"], field)
        self.assertEqual(ids["forget"], ids["correction"])
        self.assertEqual(ids["forget"], ids["retain"])
        self.assertTrue(binding["authorized_for_training"])
        self.assertFalse(binding["reviewed_for_gradients"])
        self.assertIn("not general capability", binding["loss_policy"]["retain_scope"])

    def test_qa_components_and_translations_share_one_protected_panel(self):
        binding = warbias_data.prepare(self.repo, self.output)
        split = binding["splits"]
        components = {key: set(value["components"]) for key, value in split.items()}
        self.assertFalse(components["training"] & components["development"])
        self.assertFalse(components["evaluation"] & components["training"])
        self.assertFalse(components["evaluation"] & components["development"])
        self.assertEqual(len(components["evaluation"]), 24)
        self.assertEqual(len(components["training"] | components["development"]), 28)
        eligible = len(split["training"]["ids"]) + len(split["development"]["ids"])
        self.assertEqual(eligible, 290)
        languages = []
        for language in ("uk", "en"):
            spec = binding["evaluation"][f"triplets_{language}"]
            languages.append([row["id"] for row in read_rows(self.output / spec["path"])])
            self.assertEqual(binding["evaluation"][f"qa_{language}"]["rows"], 240)
        self.assertEqual(languages[0], languages[1])
        self.assertEqual(languages[0], split["evaluation"]["ids"])
        self.assertEqual(len(languages[0]), 717)
        self.assertEqual(set(binding["permitted_gradient_ids"]), set(split["training"]["ids"]))
        self.assertFalse(set(binding["permitted_gradient_ids"]) & set(languages[0]))

    def test_split_is_reproducible_and_seeded_without_changing_evaluation(self):
        first = warbias_data.prepare(self.repo, self.output)
        other = self.root / "other"
        self.assertEqual(warbias_data.prepare(self.repo, other), first)
        seeded = warbias_data.prepare(self.repo, self.root / "seeded", seed=7)
        self.assertNotEqual(first["splits"]["development"], seeded["splits"]["development"])
        self.assertEqual(first["splits"]["evaluation"], seeded["splits"]["evaluation"])
        overview = warbias_data.overview(self.repo)
        self.assertEqual(overview["triplets_per_language"], 1007)
        self.assertEqual(overview["split_counts"], first["split_counts"])

    def test_modified_sources_or_component_mapping_are_rejected(self):
        for name in ("triplets_uk", "profile_mapping"):
            with self.subTest(name=name):
                path = self.repo / warbias_data.SOURCES[name]["path"]
                payload = path.read_bytes()
                path.write_bytes(payload + b"\n")
                with self.assertRaisesRegex(ValueError, "source checksum differs"):
                    warbias_data.prepare(self.repo, self.output)
                self.assertFalse(self.output.exists())
                path.write_bytes(payload)

    def test_declared_hash_updates_cannot_authorize_changed_text_or_components(self):
        binding = warbias_data.prepare(self.repo, self.output)
        path = self.output / binding["streams"]["forget"]["path"]
        rows = read_rows(path)
        rows[0]["target"] = "Інший текст."
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        binding["streams"]["forget"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.output / "training_binding.json").write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, "training binding differs"):
            warbias_data.validate_bundle(self.output, self.repo)

    def test_stream_changes_and_role_swaps_are_rejected_with_original_binding(self):
        binding = warbias_data.prepare(self.repo, self.output)
        path = self.output / binding["streams"]["forget"]["path"]
        path.write_bytes((self.output / binding["streams"]["correction"]["path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "source role or split"):
            warbias_data.validate_bundle(self.output, self.repo)

    def test_forged_component_allocation_cannot_allow_evaluation_for_gradients(self):
        binding = warbias_data.prepare(self.repo, self.output)
        evaluation_id = binding["locked_evaluation_ids"].pop()
        binding["permitted_gradient_ids"].append(evaluation_id)
        (self.output / "training_binding.json").write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, "training binding differs"):
            warbias_data.validate_bundle(self.output, self.repo)

    def test_returned_metadata_cannot_modify_the_source_pins(self):
        binding = warbias_data.prepare(self.repo, self.output)
        pinned = warbias_data.SOURCES["triplets_uk"]["sha256"]
        binding["source"]["files"]["triplets_uk"]["sha256"] = "changed"
        self.assertEqual(warbias_data.SOURCES["triplets_uk"]["sha256"], pinned)
        self.assertEqual(
            warbias_data.validate_bundle(self.output, self.repo)["source"]["files"]["triplets_uk"][
                "sha256"
            ],
            pinned,
        )

    def test_duplicate_source_ids_and_language_misalignment_are_rejected(self):
        for mode in ("duplicate", "alignment"):
            with self.subTest(mode=mode):
                specs = copy.deepcopy(warbias_data.SOURCES)
                path = self.repo / specs["triplets_en"]["path"]
                payload = path.read_bytes()
                rows = read_rows(path)
                if mode == "duplicate":
                    rows[-1]["id"] = rows[0]["id"]
                    error = "Duplicate source IDs"
                else:
                    rows[0], rows[1] = rows[1], rows[0]
                    error = "IDs or order differ"
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                specs["triplets_en"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with mock.patch.object(warbias_data, "SOURCES", specs):
                    with self.assertRaisesRegex(ValueError, error):
                        warbias_data.prepare(self.repo, self.output)
                self.assertFalse(self.output.exists())
                path.write_bytes(payload)

    def test_existing_bundle_and_invalid_allocation_are_not_written(self):
        warbias_data.prepare(self.repo, self.output)
        original = (self.output / "training_binding.json").read_bytes()
        with self.assertRaises(FileExistsError):
            warbias_data.prepare(self.repo, self.output)
        self.assertEqual((self.output / "training_binding.json").read_bytes(), original)
        for fraction in (0, 1, float("nan"), True):
            target = self.root / "invalid"
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "development fraction"):
                    warbias_data.prepare(self.repo, target, development_fraction=fraction)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
