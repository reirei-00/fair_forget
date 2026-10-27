import json
import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_DIR / "notebooks" / "not_executed" / "winobias_translation_comparison_200.ipynb"
)


class WinoBiasNotebookTests(unittest.TestCase):
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

    def test_source_notebook_is_clean(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_code_cells_are_valid_python(self):
        for index, source in enumerate(self.code_cells):
            compile(self.without_magics(source), f"cell_{index}", "exec")

    def test_cell_ids_are_unique(self):
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

    def test_no_api_key_is_embedded(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        self.assertNotIn('LAPA_API_KEY = "', self.code)
        self.assertNotIn('DEEPL_AUTH_KEY = "', self.code)
        self.assertNotIn('GEMINI_API_KEY = "', self.code)

    def test_canonical_sources_are_pinned(self):
        self.assertIn('SOURCE_COMMIT = "0bce984dd081bbc10b0622f326727a024c607895"', self.code)
        self.assertIn("raw.githubusercontent.com/uclanlp/corefBias", self.code)
        self.assertEqual(self.code.count("_stereotyped_type"), 16)
        self.assertEqual(self.code.count("Expected 396 rows in {filename}"), 1)
        self.assertIn("len(frame) != 3168", self.code)
        self.assertIn("int(valid_pairs.sum()) != 1582", self.code)
        self.assertIn("Source integrity check failed", self.code)

    def test_sample_keeps_complete_pairs(self):
        self.assertIn("SAMPLE_SIZE = 200", self.code)
        self.assertIn("HUMAN_REVIEW_SIZE = 40", self.code)
        self.assertIn("PAIR_ALLOCATION", self.code)
        self.assertIn("HUMAN_PAIR_ALLOCATION", self.code)
        self.assertIn('frame["counterfactual_pair_valid"]', self.code)
        self.assertIn('sample.groupby("pair_id").size().eq(2).all()', self.code)
        self.assertIn('{"anti": 100, "pro": 100}', self.code)
        self.assertIn('{"female": 100, "male": 100}', self.code)

    def test_prompt_preserves_coreference_structure(self):
        prompt = self.cells_by_id["0c3f8b51"]
        self.assertIn("Translate the complete WinoBias item", prompt)
        self.assertIn("Preserve square brackets", prompt)
        self.assertIn("pronoun gender", prompt)
        self.assertIn("pro/anti-stereotypical condition", prompt)
        self.assertIn("exactly one key: sentence", prompt)
        self.assertIsNone(re.search(r"[А-Яа-яІіЇїЄєҐґ]", prompt))

    def test_same_three_translation_systems_are_used(self):
        self.assertIn('SYSTEMS = ["deepl", "gemini", "lapa"]', self.code)
        self.assertIn('"gemini-3.5-flash-lite"', self.code)
        self.assertIn('"LapaLLM-Gemma-3-12B-instruct"', self.code)
        translation_helpers = self.cells_by_id["9f23d114"]
        self.assertEqual(translation_helpers.count("client.chat.completions.create("), 1)
        self.assertIn("translation_messages(row)", self.code)
        self.assertIn("context=deepl_item_context(row)", self.code)

    def test_deepl_preserves_annotated_spans(self):
        deepl_cell = self.cells_by_id["bd2cce5c"]
        self.assertIn('replace("[", "<coref>")', deepl_cell)
        self.assertIn('tag_handling="xml"', deepl_cell)
        self.assertIn('replace("<coref>", "[")', deepl_cell)
        self.assertIn("model_type=deepl.ModelType.PREFER_QUALITY_OPTIMIZED", deepl_cell)
        self.assertIn("custom_instructions=DEEPL_CUSTOM_INSTRUCTIONS", deepl_cell)
        self.assertIn('non_splitting_tags=["coref"]', deepl_cell)
        self.assertIn('tag_handling_version="v2"', deepl_cell)
        self.assertIn("restore_deepl_pronoun_annotations", deepl_cell)
        self.assertIn("restore_deepl_possessive_annotation", deepl_cell)
        self.assertIn("restore_deepl_occupation_annotation", deepl_cell)
        self.assertIn("merge_adjacent_annotations", deepl_cell)

    def test_results_resume_and_export_structural_failures(self):
        self.assertIn('TRANSLATIONS_PATH = OUTPUT_DIR / "translations_wide.csv"', self.code)
        self.assertIn("Saved translations contain items outside", self.code)
        self.assertIn("Complete every provider translation", self.code)
        self.assertIn('checks["benchmark_structure_valid"]', self.code)
        self.assertNotIn("Resolve changed coreference annotations before scoring", self.code)
        self.assertIn("Resolve missing translations before scoring", self.code)

    def test_invalid_outputs_are_corrected_during_translation(self):
        prompt = self.cells_by_id["0c3f8b51"]
        helpers = self.cells_by_id["9f23d114"]
        validation = self.cells_by_id["c7c021d0"]
        cache = self.cells_by_id["7943b8f1"]
        self.assertIn("STRUCTURE_CORRECTION_PROMPT", prompt)
        self.assertIn("translation_correction_messages", helpers)
        self.assertIn("validate_benchmark_translation(row, corrected)", helpers)
        self.assertIn("validate_benchmark_translation", validation)
        self.assertIn("cached_translation_valid", cache)
        self.assertIn('translations.loc[index, columns] = ""', cache)

    def test_output_structure_matches_translation_comparison(self):
        expected = [
            "sample_200.csv",
            "human_review_40.csv",
            "translations_wide.csv",
            "translation_comparison_200.csv",
            "structural_checks.csv",
            "translation_segments.csv",
            "automatic_metrics.csv",
            "automatic_metrics_summary.csv",
            "paired_bootstrap.csv",
            "metric_distributions.png",
            "metric_ranks.png",
        ]
        for filename in expected:
            self.assertIn(filename, self.code)
        self.assertIn("winobias_mt_comparison_200_artifacts.zip", self.code)

    def test_same_metrics_and_plots_are_present(self):
        self.assertIn('COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"', self.code)
        self.assertIn('METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"', self.code)
        self.assertIn('LABSE_MODEL = "sentence-transformers/LaBSE"', self.code)
        self.assertIn("Rank (1 = strongest)", self.code)
        self.assertIn('segments.groupby(["item_id", "system"])', self.code)


if __name__ == "__main__":
    unittest.main()
