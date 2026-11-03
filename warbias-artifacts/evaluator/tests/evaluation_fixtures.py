"""Synthetic tokenizer and backend for collection checks."""

import json
from types import SimpleNamespace


class FakeTokenizer:
    def __init__(self):
        self.state = {"model": {"vocab": {"a": 1}}, "normalizer": {"type": "fixture"}}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(self.state))
        self.bos_token_id, self.pad_token_id, self.eos_token_id = 10, 11, 12
        self.special_tokens_map = {
            "bos_token": "<bos>",
            "pad_token": "<pad>",
            "eos_token": "<eos>",
        }
        self.all_special_ids = [10, 11, 12]
        self.chat_template = "user:{{content}}:assistant:"
        self.add_bos_token = False
        self.offset = 0
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return ([self.bos_token_id] if add_special_tokens else []) + [
            ord(c) + self.offset for c in text
        ]


class FakeBackend:
    backend = "causal"
    max_length = 10000

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chat_template_args = {"enable_thinking": False}
        self.calls = []

    def _loglikelihood_tokens(self, *_args):
        raise AssertionError("Asset identity must never request likelihoods")

    def generate_until(self, *_args):
        raise AssertionError("Asset identity must never request generation")

    def tok_encode(self, text):
        self.calls.append(text)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def apply_chat_template(self, history, add_generation_prompt=True):
        assert add_generation_prompt
        return self.tokenizer.chat_template.replace(
            "{{content}}", history[0]["content"]
        )
