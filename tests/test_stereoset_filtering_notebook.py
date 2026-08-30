import ast
import json
import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "not_executed" / "stereoset_filtering.ipynb"
TARGET_WORDS_PATH = PROJECT_DIR / "notebooks" / "stereoset_filtering" / "target_words.csv"


class StereoSetFilteringNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.cells_by_id = {
            cell["id"]: "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        }
        cls.code = "\n".join(cls.code_cells)

    @staticmethod
    def without_magics(source):
        return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))

    def test_notebook_is_clean_and_valid(self):
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell.get("source", []))
                compile(self.without_magics(source), cell["id"], "exec")
                self.assertLessEqual(len(source.splitlines()), 70)

    def test_top_level_helpers_are_unique(self):
        tree = ast.parse(self.without_magics(self.code))
        function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(len(function_names), len(set(function_names)))

    def test_no_embedded_api_key_and_no_secret_machinery(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        self.assertNotIn("read_secret", self.code)
        self.assertNotIn("load_dotenv", self.code)
        self.assertNotIn("random.seed", self.code)

    def test_official_source_is_pinned(self):
        self.assertIn('SOURCE_COMMIT = "ead7d086a64a192a1eca88e0dd2fd163de375218"', self.code)
        self.assertIn("moinnadeem/StereoSet", self.code)
        self.assertIn("4_229", self.code)
        self.assertIn('(PROJECT_DIR / "requirements.txt").exists()', self.code)

    def test_target_word_filter_uses_curated_list(self):
        self.assertIn("target_words.csv", self.code)
        self.assertIn('os.getenv("STEREOSET_TARGET_WORDS"', self.code)
        self.assertIn("def load_target_words", self.code)
        self.assertIn("def apply_target_filter", self.code)
        self.assertIn('out["target_included"]', self.code)
        self.assertTrue(TARGET_WORDS_PATH.exists())
        header = TARGET_WORDS_PATH.read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertEqual(header, "term,domain,include,comment")

    def test_every_quality_flag_is_present(self):
        expected_flags = [
            "flag_blank_extraction_failed",
            "flag_pos_mismatch",
            "flag_filler_length_gap",
            "flag_negation_only_anti_stereotype",
            "flag_target_missing_in_context",
            "flag_exact_duplicate",
            "flag_near_duplicate",
            "flag_sentence_length_gap",
            "flag_filler_frequency_gap",
            "flag_us_specific",
            "flag_topic_inconsistency",
        ]
        for flag in expected_flags:
            self.assertIn(flag, self.code)
        self.assertIn("FLAG_COLUMNS", self.code)
        self.assertIn('flags["any_flag"]', self.code)

    def test_pos_check_uses_spacy_on_full_sentences(self):
        self.assertIn('load_spacy_model("en_core_web_sm")', self.code)
        self.assertIn('alignment_mode="expand"', self.code)
        self.assertIn("span.root.pos_", self.code)
        self.assertIn("NLP.pipe(", self.code)
        self.assertNotIn("import nltk", self.code)

    def test_review_roundtrip_is_strict(self):
        self.assertIn("def build_review_frame", self.code)
        self.assertIn("def load_reviewed", self.code)
        self.assertIn('validate="one_to_one"', self.code)
        self.assertIn('{"keep", "drop"}', self.code)
        self.assertIn("stereoset_filter_review_completed.csv", self.code)

    def test_output_structure_is_separate_and_complete(self):
        self.assertIn('PROJECT_DIR / "outputs" / "stereoset_filtering"', self.code)
        self.assertIn('os.getenv("STEREOSET_FILTER_OUTPUT_DIR"', self.code)
        expected = [
            "population_flags.csv",
            "flag_summary.csv",
            "stereoset_filter_review.csv",
            "stereoset_filter_review_completed.csv",
            "stereoset_ua_candidates.csv",
            "stereoset_filter_summary.csv",
            "filtering_overview.png",
            "cluster_counts.png",
            "pos_coverage.png",
        ]
        for filename in expected:
            self.assertIn(filename, self.code)
        self.assertIn("target_terms_catalog.csv", self.code)
        self.assertIn('flags["target_cluster"]', self.code)

    def test_exports_use_repo_csv_conventions(self):
        to_csv_calls = re.findall(r"to_csv\([^)]*\)", self.code, flags=re.DOTALL)
        self.assertTrue(to_csv_calls)
        for call in to_csv_calls:
            self.assertIn("index=False", call)
            self.assertIn('encoding="utf-8-sig"', call)


if __name__ == "__main__":
    unittest.main()
