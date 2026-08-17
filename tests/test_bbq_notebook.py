import json
import re
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "not_executed" / "bbq_translation_comparison_200.ipynb"


class BBQNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n".join(cls.code_cells)

    @staticmethod
    def without_magics(source):
        return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))

    def test_notebook_has_no_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_code_cells_are_valid_and_focused(self):
        for index, source in enumerate(self.code_cells):
            compile(self.without_magics(source), f"bbq_cell_{index}", "exec")
            self.assertLessEqual(len(source.splitlines()), 70)

    def test_cell_ids_are_unique(self):
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))

    def test_no_embedded_api_key(self):
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{12,}", self.text))
        self.assertIsNone(re.search(r"AQ\.[A-Za-z0-9_-]{20,}", self.text))
        self.assertNotIn('LAPA_API_KEY = "', self.code)
        self.assertNotIn('DEEPL_AUTH_KEY = "', self.code)
        self.assertNotIn('GEMINI_API_KEY = "', self.code)

    def test_local_and_colab_paths_are_supported(self):
        self.assertIn("load_dotenv", self.code)
        self.assertIn("from google.colab import userdata", self.code)
        self.assertIn('(PROJECT_DIR / "requirements.txt").exists()', self.code)
        self.assertIn("BBQ_TRANSLATION_OUTPUT_DIR", self.code)
        self.assertIn('PROJECT_DIR / "outputs" / "bbq"', self.code)

    def test_pinned_bbq_population_is_loaded(self):
        self.assertIn('SOURCE_COMMIT = "bea11bd97d79217245b5871acd247b9d6eb24598"', self.code)
        self.assertIn("nyu-mll/BBQ", self.code)
        self.assertIn("len(frame) != 58_492", self.code)
        for category in (
            "Age",
            "Disability_status",
            "Gender_identity",
            "Nationality",
            "Physical_appearance",
            "Race_ethnicity",
            "Race_x_SES",
            "Race_x_gender",
            "Religion",
            "SES",
            "Sexual_orientation",
        ):
            self.assertIn(f'"{category}.jsonl"', self.code)

    def test_sample_preserves_complete_context_pairs(self):
        self.assertIn("SAMPLE_SIZE = 200", self.code)
        self.assertIn("SAMPLE_PAIR_SIZE = 100", self.code)
        self.assertIn("PILOT_SIZE = 40", self.code)
        self.assertIn("PILOT_PAIR_SIZE = 20", self.code)
        self.assertIn('frame.groupby("pair_id")', self.code)
        self.assertIn('["context_condition"].nunique().eq(2)', self.code)
        self.assertIn('"ambig": 0, "disambig": 1', self.code)
        self.assertIn("pilot_pair_ids.issubset", self.code)

    def test_sample_is_balanced_and_deterministic(self):
        self.assertIn("PILOT_ALLOCATION", self.code)
        self.assertIn("SAMPLE_ALLOCATION", self.code)
        self.assertIn("stable_rank", self.code)
        self.assertIn("hashlib.sha256", self.code)
        self.assertIn('("Race_x_gender", "neg"): 12', self.code)
        self.assertIn('("Race_x_gender", "nonneg"): 13', self.code)

    def test_all_systems_receive_bbq_context(self):
        self.assertIn("translation_messages(row)", self.code)
        self.assertIn("context=deepl_item_context(row)", self.code)
        self.assertIn('"context_condition"', self.code)
        self.assertIn('"question_polarity"', self.code)
        self.assertIn('"answer_info"', self.code)
        self.assertIn('"correct_answer_index"', self.code)
        self.assertIn('"unknown_answer_index"', self.code)

    def test_shared_prompt_preserves_bbq_function(self):
        self.assertIn("Translate the complete BBQ item", self.code)
        self.assertIn("Translate every unknown-role answer into explicit Ukrainian", self.code)
        self.assertIn("Preserve whether the context is ambiguous or disambiguated", self.code)
        self.assertIn("context, question, ans0, ans1, ans2", self.code)
        self.assertIn("temperature=0.0", self.code)

    def test_same_three_provider_models_are_used(self):
        self.assertIn('SYSTEMS = ["deepl", "gemini", "lapa"]', self.code)
        self.assertIn('"gemini-3.5-flash-lite"', self.code)
        self.assertIn('"LapaLLM-Gemma-3-12B-instruct"', self.code)
        self.assertEqual(self.code.count("client.chat.completions.create("), 1)
        self.assertIn("deepl.ModelType.PREFER_QUALITY_OPTIMIZED", self.code)

    def test_deferred_api_rows_are_retried(self):
        self.assertIn("stop_after_attempt(6)", self.code)
        self.assertIn("wait_exponential(min=5, max=60)", self.code)
        self.assertIn("retry_if_exception(retryable_api_error)", self.code)
        self.assertIn('"perday" not in str(error).casefold()', self.code)
        self.assertIn("daily quota exhausted; rerun after the provider reset", self.code)
        self.assertIn("while pending:", self.code)
        self.assertIn("pending.append(index)", self.code)
        self.assertIn("queued for retry", self.code)

    def test_same_metrics_and_plots_are_exported(self):
        self.assertIn('COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"', self.code)
        self.assertIn('METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"', self.code)
        self.assertIn('LABSE_MODEL = "sentence-transformers/LaBSE"', self.code)
        self.assertIn("metric_distributions.png", self.code)
        self.assertIn("metric_ranks.png", self.code)
        self.assertIn('frame.groupby(["pair_id", "system"])', self.code)

    def test_same_csv_artifact_structure_is_exported(self):
        for artifact in (
            "sample_200.csv",
            "human_review_40.csv",
            "translations_wide.csv",
            "translation_comparison_200.csv",
            "structural_checks.csv",
            "translation_segments.csv",
            "automatic_metrics.csv",
            "automatic_metrics_summary.csv",
            "paired_bootstrap.csv",
        ):
            self.assertIn(artifact, self.code)
        self.assertIn("bbq_mt_comparison_200_artifacts.zip", self.code)

    def test_annotation_workflow_is_absent(self):
        for token in ("build_annotation_files", "krippendorff", "grades_", "ranking_"):
            self.assertNotIn(token, self.code)


if __name__ == "__main__":
    unittest.main()
