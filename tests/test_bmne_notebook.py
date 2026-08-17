import json
import re
import unittest
from pathlib import Path

import nbformat

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_DIR / "notebooks" / "not_executed" / "bmne_stereoset_translation_review.ipynb"
)


class BMNENotebookTests(unittest.TestCase):
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

    def test_notebook_is_clean_valid_and_focused(self):
        nbformat.validate(nbformat.from_dict(self.notebook))
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell.get("source", []))
            compile(self.without_magics(source), f"bmne_cell_{index}", "exec")
            self.assertLessEqual(len(source.splitlines()), 70)

    def test_no_api_key_is_embedded(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        for name in ("DEEPL_AUTH_KEY", "GEMINI_API_KEY", "LAPA_API_KEY"):
            self.assertNotIn(f'{name} = "', self.code)

    def test_pinned_bmne_source_is_verified(self):
        self.assertIn('SOURCE_COMMIT = "18a8cc7118adf65f3a293c44b69444784d095167"', self.code)
        self.assertIn("teias-ai/BMNE", self.code)
        self.assertIn("8408842bbafeec31fcd6e274dad9a8495df330fc2eaf7bf456b116dc8831b64b", self.code)
        self.assertIn("SAMPLE_SIZE = 223", self.code)
        for category, count in (
            ("Attitudes and Beliefs", 88),
            ("Roles and Behaviors", 49),
            ("Physical Characteristics", 28),
            ("Personality Traits", 58),
        ):
            self.assertIn(f'"{category}": {count}', self.code)

    def test_review_subset_is_category_balanced(self):
        self.assertIn("PILOT_SIZE = 40", self.code)
        self.assertIn("PILOT_PER_CATEGORY = 10", self.code)
        self.assertIn("stable_rank", self.code)
        self.assertIn("hashlib.sha256", self.code)
        self.assertIn('groupby("category"', self.code)
        self.assertIn("The BMNE review subset is not category-balanced", self.code)

    def test_all_three_systems_translate_complete_pairs(self):
        self.assertIn('SYSTEMS = ["deepl", "gemini", "lapa"]', self.code)
        self.assertIn('FIELDS = ["sent_less", "sent_more"]', self.code)
        self.assertIn("BMNE_GEMINI_MODEL", self.code)
        self.assertIn('"gemini-3.1-flash-lite"', self.code)
        self.assertIn('"LapaLLM-Gemma-3-12B-instruct"', self.code)
        self.assertEqual(self.code.count("client.chat.completions.create("), 1)
        self.assertIn("translation_messages(row)", self.code)
        self.assertIn("context=deepl_item_context(row)", self.code)

    def test_shared_prompt_preserves_pair_structure(self):
        prompt = self.cells_by_id["0c3f8b51"]
        self.assertIn("Translate the complete BMNE StereoSet pair", prompt)
        self.assertIn("Keep the two translations as parallel", prompt)
        self.assertIn("sent_less, sent_more", prompt)
        self.assertIn("Do not culturally adapt", prompt)
        self.assertIsNone(re.search(r"[А-Яа-яІіЇїЄєҐґ]", prompt))

    def test_results_resume_after_interruption(self):
        self.assertIn('TRANSLATIONS_PATH = OUTPUT_DIR / "translations_wide.csv"', self.code)
        self.assertIn("Saved translations contain items outside", self.code)
        self.assertIn("except Exception as error", self.code)
        self.assertIn("provider_wide_failure", self.code)
        self.assertIn("provider quota unavailable", self.code)
        self.assertIn("break", self.cells_by_id["translation-runner"])
        self.assertIn("retry_if_exception", self.code)
        self.assertIn("ThreadPoolExecutor", self.code)
        self.assertIn("range(0, len(pending), 3)", self.cells_by_id["translation-runner"])
        self.assertIn("missing translations without recorded errors", self.code)
        self.assertIn("eligible_item_ids", self.code)
        self.assertIn("Fewer than 95%", self.code)

    def test_complete_metric_pipeline_is_present(self):
        self.assertIn('COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"', self.code)
        self.assertIn('METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"', self.code)
        self.assertIn('LABSE_MODEL = "sentence-transformers/LaBSE"', self.code)
        self.assertIn("paired_bootstrap_difference", self.code)
        self.assertIn("metric_distributions.png", self.code)
        self.assertIn("metric_ranks.png", self.code)
        self.assertIn("Rank (1 = strongest)", self.code)
        self.assertIn("expected_segments = len(eligible_item_ids)", self.code)

    def test_annotator_exports_are_simple_and_complete(self):
        export = self.cells_by_id["8cd7d538"]
        for token in (
            "less_stereotypical_deepl_uk",
            "less_stereotypical_gemini_uk",
            "less_stereotypical_lapa_uk",
            "less_stereotypical_uk_final",
            "more_stereotypical_deepl_uk",
            "more_stereotypical_gemini_uk",
            "more_stereotypical_lapa_uk",
            "more_stereotypical_uk_final",
            "preferred_system",
            "review_status",
            "pair_meaning_preserved",
            "missing_translation:",
        ):
            self.assertIn(token, export)
        for filename in (
            "translation_comparison_223.csv",
            "annotator_review_223.csv",
            "human_review_40.csv",
            "bmne_stereoset_mt_comparison_223_artifacts.zip",
        ):
            self.assertIn(filename, self.code)

    def test_local_and_colab_execution_are_supported(self):
        self.assertIn("load_dotenv", self.code)
        self.assertIn("from google.colab import userdata", self.code)
        self.assertIn('(PROJECT_DIR / "requirements.txt").exists()', self.code)
        self.assertIn("BMNE_OUTPUT_DIR", self.code)
        self.assertIn("files.download", self.code)


if __name__ == "__main__":
    unittest.main()
