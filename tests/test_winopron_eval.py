import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "evaluation" / "winopron_uk" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("winopron_uk_evaluate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=True):
        prefix = [2] if add_special_tokens else []
        return [*prefix, *(ord(character) for character in text)]


class FakeChatTokenizer(FakeTokenizer):
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        return f"CHAT:{messages[0]['content']}"


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeOutput:
    def __init__(self, request, answer_logprob):
        self.prompt_logprobs = [None] * len(request.prompt_token_ids)
        for position in range(request.score_start, len(request.prompt_token_ids)):
            token_id = request.prompt_token_ids[position]
            self.prompt_logprobs[position] = {token_id: FakeLogprob(answer_logprob)}


class WinoPronEvaluationTests(unittest.TestCase):
    def prompt(self, order="occupation_first", pronoun_set="male", gold_role="occupation"):
        occupation_first = order == "occupation_first"
        option_a_role = "occupation" if occupation_first else "participant"
        option_b_role = "participant" if occupation_first else "occupation"
        gold_option = "A" if option_a_role == gold_role else "B"
        row_id = f"1-2:{gold_role}:double:{pronoun_set}"
        return MODULE.Prompt(
            prompt_id=f"{row_id}:{order}",
            row_id=row_id,
            pair_id="1-2",
            context="double",
            pronoun_type="nominative",
            pronoun_set=pronoun_set,
            option_order=order,
            option_a_role=option_a_role,
            option_b_role=option_b_role,
            gold_role=gold_role,
            gold_option=gold_option,
            prompt_uk="Відповідайте лише A або B.\nВідповідь: ",
            occupation_group_uk="технік",
        )

    def prediction(self, prompt, predicted_role):
        predicted_option = next(
            option for option, role in prompt.option_roles.items() if role == predicted_role
        )
        return MODULE.PromptPrediction(
            prompt_id=prompt.prompt_id,
            row_id=prompt.row_id,
            pair_id=prompt.pair_id,
            context=prompt.context,
            pronoun_type=prompt.pronoun_type,
            pronoun_set=prompt.pronoun_set,
            option_order=prompt.option_order,
            occupation_group_uk=prompt.occupation_group_uk,
            gold_role=prompt.gold_role,
            gold_option=prompt.gold_option,
            predicted_option=predicted_option,
            predicted_role=predicted_role,
            attempted=True,
            correct=float(predicted_role == prompt.gold_role),
            score_margin=1.0,
        )

    def test_cli_defaults_use_raw_prompt(self):
        args = MODULE.parse_args(["--output-dir", "outputs"])
        self.assertEqual(args.prompt_format, "raw")

    def test_raw_prompt_does_not_apply_chat_template(self):
        tokenizer = FakeChatTokenizer()
        raw = MODULE.render_prompt(tokenizer, self.prompt(), "raw")
        chat = MODULE.render_prompt(tokenizer, self.prompt(), "chat")
        self.assertFalse(raw.startswith("CHAT:"))
        self.assertTrue(chat.startswith("CHAT:"))

    def test_only_answer_suffix_is_scored(self):
        request = MODULE.build_prompt_request(FakeTokenizer(), self.prompt(), "A")
        self.assertEqual(len(request.prompt_token_ids) - request.score_start, 1)
        mean_logprob, token_count = MODULE.extract_mean_logprob(FakeOutput(request, -0.5), request)
        self.assertEqual(mean_logprob, -0.5)
        self.assertEqual(token_count, 1)

    def test_option_orders_must_select_same_role(self):
        first = self.prompt("occupation_first")
        second = self.prompt("participant_first")
        consistent = MODULE.build_row_predictions(
            [self.prediction(first, "occupation"), self.prediction(second, "occupation")]
        )[0]
        inconsistent = MODULE.build_row_predictions(
            [self.prediction(first, "occupation"), self.prediction(second, "participant")]
        )[0]
        self.assertTrue(consistent.position_consistent)
        self.assertTrue(consistent.correct)
        self.assertFalse(inconsistent.position_consistent)
        self.assertFalse(inconsistent.correct)

    def test_pair_diagnostic_finds_occupation_overselection(self):
        prompt_predictions = []
        for gold_role in MODULE.ROLES:
            for order in MODULE.OPTION_ORDERS:
                prompt = self.prompt(order, gold_role=gold_role)
                prompt_predictions.append(self.prediction(prompt, "occupation"))
        rows = MODULE.build_row_predictions(prompt_predictions)
        diagnostic = MODULE.build_pair_diagnostics(rows)[0]
        self.assertFalse(diagnostic.pair_correct)
        self.assertEqual(diagnostic.error_direction, "occupation_overselection")

    def test_summary_reports_signed_gender_gap(self):
        prompt_predictions = []
        for pronoun_set in MODULE.PRONOUN_SETS:
            for gold_role in MODULE.ROLES:
                predicted_role = (
                    "participant"
                    if pronoun_set == "female" and gold_role == "occupation"
                    else gold_role
                )
                for order in MODULE.OPTION_ORDERS:
                    prompt = self.prompt(order, pronoun_set, gold_role)
                    prompt_predictions.append(self.prediction(prompt, predicted_role))
        rows = MODULE.build_row_predictions(prompt_predictions)
        pairs = MODULE.build_pair_diagnostics(rows)
        summary = {
            row["metric"]: row["value"]
            for row in MODULE.calculate_summary(prompt_predictions, rows, pairs)
        }
        self.assertEqual(summary["double_male_accuracy"], 100.0)
        self.assertEqual(summary["double_female_accuracy"], 50.0)
        self.assertEqual(summary["female_minus_male_accuracy"], -50.0)


if __name__ == "__main__":
    unittest.main()
