"""Offline synthetic fixtures for generation and translation examples."""

from copy import deepcopy


class FixtureProvider:
    """A deterministic local provider; no network client or credential access."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate(self, request):
        self.calls.append(deepcopy(request))
        return {
            "text": self.outputs.pop(0),
            "model": "synthetic-fixture",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


GENERATION_CONFIG = {
    "provider": "local-fixture",
    "model": "synthetic-fixture",
    "revision": "a" * 40,
    "settings": {"temperature": 0},
}
SOURCES = [
    {"source_id": "synthetic-source", "repo": "fixture/synthetic-source", "revision": "a" * 40}
]


def run_generation_example(output, *, prompt_root):
    from fairforget.examples_generation import run_generation_example as run

    return run(output, prompt_root=prompt_root)


def run_translation_example(output, *, prompt_root):
    from fairforget.examples_generation import run_translation_example as run

    return run(output, prompt_root=prompt_root)
