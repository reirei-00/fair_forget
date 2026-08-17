import ast
import json
import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "not_executed" / "stereoset_translation_comparison_200.ipynb"
WINOGENDER_NOTEBOOK_PATH = (
    PROJECT_DIR / "notebooks" / "not_executed" / "winogender_translation_comparison_200.ipynb"
)


class NotebookTests(unittest.TestCase):
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

    def test_notebook_has_no_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_no_embedded_api_key(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        self.assertNotIn('LAPA_API_KEY = "', self.code)
        self.assertNotIn('DEEPL_AUTH_KEY = "', self.code)
        self.assertNotIn('GEMINI_API_KEY = "', self.code)

    def test_code_cells_are_valid_python(self):
        for index, source in enumerate(self.code_cells):
            compile(self.without_magics(source), f"cell_{index}", "exec")

    def test_code_cells_stay_focused(self):
        for source in self.code_cells:
            self.assertLessEqual(len(source.splitlines()), 70)

    def test_cell_ids_are_unique(self):
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

    def test_top_level_helpers_are_unique(self):
        tree = ast.parse(self.without_magics(self.code))
        function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(len(function_names), len(set(function_names)))

    def test_local_and_colab_secret_support(self):
        self.assertIn("load_dotenv", self.code)
        self.assertIn("from google.colab import userdata", self.code)
        self.assertIn("def read_secret", self.code)
        self.assertIn('(PROJECT_DIR / "requirements.txt").exists()', self.code)

    def test_all_systems_receive_item_context(self):
        self.assertIn("translation_messages(row)", self.code)
        self.assertIn("context=deepl_item_context(row)", self.code)
        self.assertIn('"bias_category"', self.code)
        self.assertIn('"bias_target"', self.code)

    def test_sample_200_preserves_the_pilot(self):
        self.assertIn("PILOT_SIZE = 40", self.code)
        self.assertIn("SAMPLE_SIZE = 200", self.code)
        self.assertIn("pilot_ids.issubset", self.code)
        self.assertIn("sample_200.csv", self.code)
        self.assertIn("Saved translations contain items outside", self.code)

    def test_shared_llm_prompt_is_english(self):
        prompt = self.cells_by_id["0c3f8b51"]
        self.assertIn("Translate the complete StereoSet item", prompt)
        self.assertIsNone(re.search(r"[А-Яа-яІіЇїЄєҐґ]", prompt))
        self.assertNotIn("SHARED_SYSTEM_PROMPT_UK", self.code)
        self.assertNotIn("MODE_CONTEXT_UK", self.code)
        self.assertIn('"bias_category_description"', self.code)

    def test_chat_api_request_is_implemented_once(self):
        self.assertEqual(self.code.count("client.chat.completions.create("), 1)
        self.assertNotIn("def translation_columns", self.code)
        self.assertIn("OUTPUT_COLUMNS", self.code)
        self.assertIn("/{len(translations)} translated", self.code)
        self.assertIn("Complete every provider translation", self.code)
        self.assertNotIn("check_lapa_connection", self.code)
        self.assertNotIn("json_output", self.code)
        self.assertNotIn("temperature is not None", self.code)
        self.assertNotIn("if delay_seconds", self.code)
        self.assertNotIn("if not IN_COLAB", self.code)

    def test_gemini_uses_stable_api_model(self):
        self.assertIn('read_secret("GEMINI_API_KEY")', self.code)
        self.assertIn('"gemini-3.5-flash-lite"', self.code)
        self.assertIn("generativelanguage.googleapis.com/v1beta/openai/", self.code)
        self.assertIn('response_format={"type": "json_object"}', self.code)
        self.assertNotIn("GEMMA_BACKEND", self.code)
        self.assertNotIn('run_translator("gemma4"', self.code)
        self.assertNotIn("bitsandbytes", self.code)
        self.assertIn(
            'run_translator("gemini", translate_gemini_item, delay_seconds=5.0)',
            self.code,
        )

    def test_metricx24_qe_large_is_scored(self):
        self.assertIn('COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"', self.code)
        self.assertNotIn("wmt22-cometkiwi-da", self.code)
        self.assertIn('message=r"`resume_download` is deprecated.*"', self.code)
        self.assertIn('module=r"huggingface_hub\\.file_download"', self.code)
        self.assertIn('os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")', self.code)
        self.assertIn("num_workers=1", self.code)
        self.assertIn('METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"', self.code)
        self.assertIn('"transformers==4.30.2"', self.code)
        self.assertIn('"huggingface-hub>=0.34,<1.0"', self.code)
        self.assertIn('"numpy==1.26.4"', self.code)
        self.assertIn('"scipy==1.15.1"', self.code)
        self.assertIn("METRICX_SOURCE_REVISION", self.code)
        self.assertIn("METRICX_SOURCE_SHA256", self.code)
        self.assertIn('segments["metricx24_qe_error"]', self.code)
        self.assertIn('"metricx24_qe_error": "lower"', self.code)

    def test_labse_cosine_is_scored(self):
        self.assertIn("Higher `comet20_mean`", self.text)
        self.assertIn("Lower `mean_rank`", self.text)
        self.assertNotIn("TODO: LaBSE cosine similarity", self.text)
        self.assertIn('LABSE_MODEL = "sentence-transformers/LaBSE"', self.code)
        self.assertIn('segments["labse_cosine"]', self.code)
        self.assertIn('"labse_cosine": "higher"', self.code)
        self.assertNotIn("from sentence_transformers import", self.code)

    def test_metric_plots_are_exported(self):
        self.assertIn("metric_distributions.png", self.code)
        self.assertIn("metric_ranks.png", self.code)
        self.assertIn('segments.groupby(["item_id", "system"])', self.code)
        self.assertIn("Rank (1 = strongest)", self.code)

    def test_annotation_workflow_is_absent(self):
        forbidden = ["build_annotation_files", "krippendorff", "grades_", "ranking_"]
        for token in forbidden:
            self.assertNotIn(token, self.code)

    def test_validation_transformation_has_no_nested_loops(self):
        validation_code = self.cells_by_id["1d718b00"]
        tree = ast.parse(validation_code)
        self.assertNotIn("build_qa_tables", validation_code)
        for loop in (node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))):
            nested_loops = (
                child
                for statement in loop.body
                for child in ast.walk(statement)
                if isinstance(child, (ast.For, ast.While))
            )
            self.assertFalse(any(nested_loops))

    def test_processing_csv_is_exported(self):
        self.assertIn("translation_comparison_200.csv", self.code)


class WinoGenderNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(WINOGENDER_NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.text = WINOGENDER_NOTEBOOK_PATH.read_text(encoding="utf-8")
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
                self.assertLessEqual(len(source.splitlines()), 85)

    def test_no_embedded_api_key(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        self.assertNotIn('LAPA_API_KEY = "', self.code)
        self.assertNotIn('DEEPL_AUTH_KEY = "', self.code)
        self.assertNotIn('GEMINI_API_KEY = "', self.code)

    def test_official_source_is_pinned_and_grouped(self):
        self.assertIn("rudinger/winogender-schemas", self.code)
        self.assertIn("1c7f8b481ad8a234b41e9f76a424d6e856e13f7f", self.code)
        self.assertIn("all_sentences.tsv", self.code)
        self.assertIn('FIELDS = ["male", "female", "neutral"]', self.code)
        self.assertIn(r'r"\b(he|his|him)\b"', self.code)
        self.assertIn("Expected 240 WinoGender triplets", self.code)

    def test_sample_and_review_subset_are_balanced(self):
        self.assertIn("SAMPLE_SIZE = 200", self.code)
        self.assertIn("PILOT_SIZE = 40", self.code)
        self.assertIn("SAMPLE_PER_STRATUM = 50", self.code)
        self.assertIn("PILOT_PER_STRATUM = 10", self.code)
        self.assertIn("40 distinct occupations", self.code)
        self.assertIn("sample_200.csv", self.code)

    def test_all_systems_receive_coreference_context(self):
        prompt = self.cells_by_id["0c3f8b51"]
        self.assertIn("Translate the complete WinoGender triplet", prompt)
        self.assertIsNone(re.search(r"[А-Яа-яІіЇїЄєҐґ]", prompt))
        self.assertIn('"correct_antecedent_role"', self.code)
        self.assertIn('"correct_antecedent"', self.code)
        self.assertIn('"pronoun_case"', self.code)
        self.assertIn("translation_messages(row)", self.code)
        self.assertIn("context=deepl_item_context(row)", self.code)
        self.assertNotIn("translated gender variants are not distinct", self.code)
        self.assertIn('checks["variants_distinct"]', self.code)

    def test_same_three_translation_models_are_used(self):
        self.assertEqual(self.code.count("client.chat.completions.create("), 1)
        self.assertIn("retry_if_exception(retryable_api_error)", self.code)
        self.assertIn("daily quota reached; remaining items stay pending", self.code)
        self.assertIn('"gemini-3.5-flash-lite"', self.code)
        self.assertIn('"LapaLLM-Gemma-3-12B-instruct"', self.code)
        self.assertIn("deepl.DeepLClient", self.code)
        for system in ("deepl", "gemini", "lapa"):
            self.assertIn(f'run_translator("{system}"', self.code)

    def test_output_structure_is_separate_and_complete(self):
        self.assertIn('PROJECT_DIR / "outputs" / "winogender"', self.code)
        self.assertIn('os.getenv("WINOGENDER_OUTPUT_DIR")', self.code)
        expected = [
            "translations_wide.csv",
            "translation_comparison_200.csv",
            "human_review_40.csv",
            "structural_checks.csv",
            "translation_segments.csv",
            "automatic_metrics.csv",
            "automatic_metrics_summary.csv",
            "paired_bootstrap.csv",
            "metric_distributions.png",
            "metric_ranks.png",
            "winogender_mt_comparison_200_artifacts.zip",
        ]
        for filename in expected:
            self.assertIn(filename, self.code)

    def test_same_metrics_and_plots_are_used(self):
        self.assertIn('COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"', self.code)
        self.assertIn('METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"', self.code)
        self.assertIn('LABSE_MODEL = "sentence-transformers/LaBSE"', self.code)
        self.assertIn('segments["comet20_qe"]', self.code)
        self.assertIn('segments["metricx24_qe_error"]', self.code)
        self.assertIn('segments["labse_cosine"]', self.code)
        self.assertIn("Rank (1 = strongest)", self.code)


if __name__ == "__main__":
    unittest.main()
