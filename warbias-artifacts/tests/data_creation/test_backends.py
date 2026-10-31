"""No live runtime: exact native/raw prompt and pinned vLLM call contracts."""

import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from warbias_kit.backends import vllm_provider
from warbias_kit.io import digest, read_rows
from warbias_kit.warbias.provider import request_record, run_requests


def configuration():
    return {
        "provider": "vllm",
        "model": "owner/model",
        "revision": "a" * 40,
        "settings": {
            "engine": {
                "tokenizer": "owner/tokenizer",
                "tokenizer_revision": "b" * 40,
                "dtype": "bfloat16",
                "tensor_parallel_size": 2,
                "max_model_len": 64,
                "seed": 31,
                "trust_remote_code": False,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.7,
            },
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 8,
                "seed": 32,
                "n": 1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "stop": None,
                "stop_token_ids": None,
                "ignore_eos": False,
                "detokenize": True,
                "skip_special_tokens": True,
                "spaces_between_special_tokens": False,
            },
            "prompt_format": "native-chat",
            "chat_template": {
                "source": "tokenizer",
                "add_generation_prompt": True,
                "kwargs": {"enable_thinking": False},
            },
        },
    }


@dataclass
class TokenLogprob:
    logprob: float
    rank: int
    decoded_token: str


@pytest.fixture
def runtime(monkeypatch):
    observed = {"llm": [], "sampling": [], "templates": [], "raw_encode": [], "generate": []}
    tokenizer = SimpleNamespace()
    tokenizer.get_chat_template = lambda: "{{ messages }} native template"

    def apply(messages, **kwargs):
        observed["templates"].append((deepcopy(messages), deepcopy(kwargs)))
        return [7, 8, 9]

    def encode(text, **kwargs):
        observed["raw_encode"].append((text, kwargs))
        return [1, 2]

    tokenizer.apply_chat_template = apply
    tokenizer.encode = encode

    class SamplingParams:
        def __init__(self, **kwargs):
            observed["sampling"].append(kwargs)

        def __repr__(self):
            return "SamplingParams(fixture)"

    class LLM:
        def __init__(self, **kwargs):
            observed["llm"].append(deepcopy(kwargs))
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(**kwargs))

        def get_tokenizer(self):
            return tokenizer

        def generate(self, prompts, sampling, **kwargs):
            observed["generate"].append((deepcopy(prompts), sampling, kwargs))
            return [
                SimpleNamespace(
                    request_id="fixture-request",
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            text='  {"uk":"Текст"}\n',
                            token_ids=[4, 5],
                            finish_reason="stop",
                            stop_reason=5,
                            cumulative_logprob=-0.7,
                            logprobs=[
                                {4: TokenLogprob(-0.3, 1, "Текст")},
                                {5: TokenLogprob(-0.4, 2, "!")},
                            ],
                        )
                    ],
                    finished=True,
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams, __version__="0.18.0-fixture"),
    )
    return observed


def request(config):
    return request_record(
        "item",
        "translation",
        {"source": "An exact source"},
        config,
        {"text": "Exact system prompt", "sha256": "c" * 64},
    )


def test_structured_output_uses_the_declared_stage_schema(runtime, monkeypatch):
    @dataclass
    class StructuredOutputsParams:
        json: dict

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(StructuredOutputsParams=StructuredOutputsParams),
    )
    config = configuration()
    schema = {
        "type": "object",
        "properties": {"uk": {"type": "string"}},
        "required": ["uk"],
        "additionalProperties": False,
    }
    config["settings"]["response_schemas"] = {"translation": schema}
    response = vllm_provider(config=config).generate(request(config))
    assert runtime["sampling"][-1]["structured_outputs"].json == schema
    assert response["response_schema_sha256"] == digest(schema)
    assert response["provider_config"]["settings"]["response_schemas"] == {"translation": schema}


def test_missing_stage_schema_prevents_inference(runtime, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(StructuredOutputsParams=lambda **kwargs: kwargs),
    )
    config = configuration()
    config["settings"]["response_schemas"] = {"S": {"type": "object"}}
    with pytest.raises(ValueError, match="stage has no declared JSON schema"):
        vllm_provider(config=config).generate(request(config))
    assert not runtime["generate"]


def test_native_adapter_preserves_exact_settings_messages_and_raw_evidence(runtime, tmp_path):
    config = configuration()
    config["unused_api_key"] = "must-not-be-retained-by-local-backend"
    provider = vllm_provider(config=config)
    record = request(config)
    before = deepcopy(record)
    rows = run_requests([record], tmp_path / "run", provider=provider, execute=True)
    response = rows[0]["response"]
    assert record == before
    assert runtime["llm"] == [
        {"model": config["model"], "revision": config["revision"], **config["settings"]["engine"]}
    ]
    assert runtime["sampling"] == [config["settings"]["sampling"]]
    assert runtime["templates"] == [
        (
            record["messages"],
            {
                "chat_template": "{{ messages }} native template",
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]
    prompts, _params, kwargs = runtime["generate"][0]
    assert prompts == [{"prompt_token_ids": [7, 8, 9]}] and kwargs == {"use_tqdm": False}
    assert response["text"] == '  {"uk":"Текст"}\n'
    assert response["model_identity"] == {
        "model": "owner/model",
        "revision": "a" * 40,
        "tokenizer": "owner/tokenizer",
        "tokenizer_revision": "b" * 40,
    }
    assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert response["input"]["prompt_tokens_sha256"] == digest([7, 8, 9])
    assert response["raw"]["outputs"][0]["logprobs"][0]["4"] == {
        "logprob": -0.3,
        "rank": 1,
        "decoded_token": "Текст",
    }
    assert "unused_api_key" not in response["provider_config"]
    saved = read_rows(tmp_path / "run/responses.jsonl")[0]
    assert saved["response"] == response
    run_requests([record], tmp_path / "run", provider=provider, execute=True)
    assert len(runtime["generate"]) == 1


def test_explicit_literal_template_is_forwarded_without_tokenizer_default(runtime):
    config = configuration()
    config["settings"]["chat_template"].update(source="literal", text="An exact custom template")
    response = vllm_provider(config=config).generate(request(config))
    assert runtime["templates"][0][1]["chat_template"] == "An exact custom template"
    assert response["input"]["chat_template"] == "An exact custom template"


def test_raw_prompt_never_guesses_message_concatenation_or_special_tokens(runtime):
    config = configuration()
    config["settings"].pop("chat_template")
    config["settings"].update(prompt_format="raw", add_special_tokens=False)
    provider = vllm_provider(config=config)
    record = request(config)
    with pytest.raises(ValueError, match="one user text"):
        provider.generate(record)
    record["messages"] = [{"role": "user", "content": " raw text\n"}]
    response = provider.generate(record)
    assert runtime["raw_encode"] == [(" raw text\n", {"add_special_tokens": False})]
    assert response["input"]["prompt_token_ids"] == [1, 2]
    assert not runtime["templates"]


@pytest.mark.parametrize("target", ["model", "tokenizer", "settings", "template"])
def test_missing_or_mutable_choices_fail_before_loading(monkeypatch, target):
    config = configuration()
    if target == "model":
        config["revision"] = "main"
    elif target == "tokenizer":
        config["settings"]["engine"]["tokenizer_revision"] = "latest"
    elif target == "template":
        config["settings"].pop("chat_template")
    else:
        config["settings"]["sampling"].pop("seed")
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ValueError):
        vllm_provider(config=config)


def test_changed_request_and_context_overflow_never_reach_generation(runtime):
    config = configuration()
    provider = vllm_provider(config=config)
    record = request(config)
    record["settings"]["sampling"]["temperature"] = 0.9
    with pytest.raises(ValueError, match="loaded provider"):
        provider.generate(record)
    config["settings"]["engine"]["max_model_len"] = 10
    provider = vllm_provider(config=config)
    with pytest.raises(ValueError, match="context size"):
        provider.generate(request(config))
    assert not runtime["generate"]


@pytest.mark.parametrize("model", ["../weights", "./weights", "/local/weights"])
def test_local_paths_cannot_claim_remote_revision_pinning(model, monkeypatch):
    config = configuration()
    config["model"] = model
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ValueError, match="Hub repository"):
        vllm_provider(config=config)
