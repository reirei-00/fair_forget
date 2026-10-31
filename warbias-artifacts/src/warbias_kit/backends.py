"""Explicit local generation adapters; importing this module never loads a model.

The vLLM adapter uses the v0.18+ token-prompt API. Every provider choice is kept
inside request-bound ``settings`` so resume cannot reuse a different protocol.
"""

import importlib
import math
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path

from warbias_kit.io import canonical_json, digest

ENGINE_REQUIRED = {
    "tokenizer",
    "tokenizer_revision",
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "seed",
    "trust_remote_code",
    "enforce_eager",
    "gpu_memory_utilization",
}
ENGINE_ALLOWED = ENGINE_REQUIRED | {
    "tokenizer_mode",
    "quantization",
    "download_dir",
    "limit_mm_per_prompt",
}
SAMPLING_REQUIRED = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "detokenize",
    "skip_special_tokens",
    "spaces_between_special_tokens",
}


def _pin(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{name} must be an immutable 40-character Git revision")


def _repository(value, name):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) is None
        or any(part in {".", ".."} for part in value.split("/"))
        or Path(value).exists()
    ):
        raise ValueError(f"{name} must identify a pinned Hub repository, not a mutable local path")


def validate_vllm_config(config):
    """Validate pinned model, tokenizer, prompt and sampling settings without loading a model."""
    config = deepcopy(config)
    if config.get("provider") != "vllm":
        raise ValueError("The vLLM factory requires provider='vllm'")
    _repository(config.get("model"), "Model")
    _pin(config.get("revision"), "Model revision")
    settings = config.get("settings")
    if not isinstance(settings, dict) or set(settings) - {
        "engine",
        "sampling",
        "prompt_format",
        "chat_template",
        "add_special_tokens",
        "response_schemas",
    }:
        raise ValueError("Declare engine, sampling and prompt protocol under settings")
    engine, sampling = settings.get("engine"), settings.get("sampling")
    schemas = settings.get("response_schemas")
    if schemas is not None and (
        not isinstance(schemas, dict)
        or not schemas
        or any(not isinstance(stage, str) or not stage for stage in schemas)
        or any(
            not isinstance(schema, dict) or schema.get("type") != "object"
            for schema in schemas.values()
        )
    ):
        raise ValueError("Declare a JSON object schema for each constrained response stage")
    if (
        not isinstance(engine, dict)
        or not ENGINE_REQUIRED <= set(engine)
        or set(engine) - ENGINE_ALLOWED
    ):
        raise ValueError("Engine settings contain pending, unsupported or missing explicit choices")
    _repository(engine["tokenizer"], "Tokenizer")
    _pin(engine["tokenizer_revision"], "Tokenizer revision")
    if engine["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Declare an explicit inference dtype")
    for key in ("tensor_parallel_size", "max_model_len"):
        if type(engine[key]) is not int or engine[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(engine["seed"]) is not int or any(
        type(engine[key]) is not bool for key in ("trust_remote_code", "enforce_eager")
    ):
        raise ValueError("Engine seed and execution choices must be explicit")
    memory = engine["gpu_memory_utilization"]
    if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("Declare a valid GPU memory fraction")
    if not isinstance(sampling, dict) or not SAMPLING_REQUIRED <= set(sampling):
        raise ValueError("Sampling settings contain pending or missing explicit choices")
    if sampling["n"] != 1 or type(sampling["n"]) is not int or sampling["detokenize"] is not True:
        raise ValueError("The text-record protocol requires exactly one detokenized output")
    if (
        type(sampling["max_tokens"]) is not int
        or sampling["max_tokens"] <= 0
        or type(sampling["seed"]) is not int
    ):
        raise ValueError("Generation needs an explicit seed and positive output limit")
    if settings.get("prompt_format") == "native-chat":
        template = settings.get("chat_template")
        if (
            not isinstance(template, dict)
            or template.get("source") not in {"tokenizer", "literal"}
            or type(template.get("add_generation_prompt")) is not bool
            or not isinstance(template.get("kwargs"), dict)
        ):
            raise ValueError(
                "Native chat requires an explicit template source, prompt flag and kwargs"
            )
        if set(template["kwargs"]) & {
            "tokenize",
            "add_generation_prompt",
            "chat_template",
            "return_dict",
            "return_tensors",
        }:
            raise ValueError("Template kwargs cannot override the registered prompt protocol")
        if template["source"] == "literal" and (
            not isinstance(template.get("text"), str) or not template["text"]
        ):
            raise ValueError("A literal template requires its exact text")
        if "add_special_tokens" in settings:
            raise ValueError(
                "Native templates own special tokens; do not request a second insertion"
            )
    elif settings.get("prompt_format") == "raw":
        if type(settings.get("add_special_tokens")) is not bool or "chat_template" in settings:
            raise ValueError(
                "Raw prompts require an explicit special-token choice and no chat template"
            )
    else:
        raise ValueError("Choose prompt_format='native-chat' or 'raw' explicitly")
    # Reject nonfinite or non-JSON configuration before importing a model runtime.
    canonical_json(settings)
    return {key: config[key] for key in ("provider", "model", "revision", "settings")}


def _json(value):
    """Retain typed backend fields, including token probabilities and timing metrics."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if is_dataclass(value):
        return _json(asdict(value))
    if hasattr(value, "__struct_fields__"):
        return {key: _json(getattr(value, key)) for key in value.__struct_fields__}
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if hasattr(value, "__dict__"):
        return {key: _json(item) for key, item in vars(value).items() if not key.startswith("_")}
    raise ValueError(f"Unsupported raw vLLM field type: {type(value).__name__}")


class VllmProvider:
    def __init__(self, config, runtime):
        self.config = validate_vllm_config(config)
        self.settings = self.config["settings"]
        self.runtime = runtime
        self.structured_type = (
            importlib.import_module("vllm.sampling_params").StructuredOutputsParams
            if self.settings.get("response_schemas")
            else None
        )
        self.sampling = runtime.SamplingParams(**self.settings["sampling"])
        self.llm = runtime.LLM(
            model=self.config["model"], revision=self.config["revision"], **self.settings["engine"]
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.version = runtime.__version__
        actual = self.llm.llm_engine.model_config
        identity = {
            "model": self.config["model"],
            "revision": self.config["revision"],
            "tokenizer": self.settings["engine"]["tokenizer"],
            "tokenizer_revision": self.settings["engine"]["tokenizer_revision"],
        }
        if any(getattr(actual, key) != value for key, value in identity.items()):
            raise ValueError(
                "Loaded vLLM model/tokenizer identity differs from the frozen configuration"
            )
        self.identity = identity
        self.template = None
        if self.settings["prompt_format"] == "native-chat":
            selected = self.settings["chat_template"]
            self.template = (
                self.tokenizer.get_chat_template()
                if selected["source"] == "tokenizer"
                else selected["text"]
            )
            if not isinstance(self.template, str) or not self.template:
                raise ValueError("The selected tokenizer has no usable native chat template")

    def generate(self, request):
        for key in ("provider", "model", "revision", "settings"):
            if request.get(key) != self.config[key]:
                raise ValueError(
                    "Generation request differs from the loaded provider configuration"
                )
        sampling = self.sampling
        schema = None
        if self.structured_type is not None:
            schema = self.settings["response_schemas"].get(request.get("stage"))
            if schema is None:
                raise ValueError("The response stage has no declared JSON schema")
            sampling = self.runtime.SamplingParams(
                **self.settings["sampling"],
                structured_outputs=self.structured_type(json=schema),
            )
        messages = request.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            )
        ):
            raise ValueError("Generation requires explicit text messages")
        if self.settings["prompt_format"] == "native-chat":
            template = self.settings["chat_template"]
            tokens = self.tokenizer.apply_chat_template(
                deepcopy(messages),
                chat_template=self.template,
                tokenize=True,
                add_generation_prompt=template["add_generation_prompt"],
                **template["kwargs"],
            )
        else:
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError(
                    "Raw mode accepts one user text; message concatenation must never be guessed"
                )
            tokens = self.tokenizer.encode(
                messages[0]["content"], add_special_tokens=self.settings["add_special_tokens"]
            )
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Tokenizer must return a nonempty list of token IDs")
        if (
            len(tokens) + self.settings["sampling"]["max_tokens"]
            > self.settings["engine"]["max_model_len"]
        ):
            raise ValueError("Prompt plus output limit exceeds the pinned context size")
        result = self.llm.generate([{"prompt_token_ids": tokens}], sampling, use_tqdm=False)
        if (
            len(result) != 1
            or len(result[0].outputs) != 1
            or list(result[0].prompt_token_ids) != tokens
        ):
            raise ValueError(
                "vLLM output coverage or effective input differs from the submitted request"
            )
        output = result[0].outputs[0]
        if not isinstance(output.text, str):
            raise ValueError("vLLM did not return raw generated text")
        raw = _json(result[0])
        response = {
            "text": output.text,
            "model": self.config["model"],
            "model_identity": dict(self.identity),
            "backend": {"name": "vllm", "version": self.version},
            "provider_config": deepcopy(self.config),
            "sampling_params_repr": repr(sampling),
            **({"response_schema_sha256": digest(schema)} if schema is not None else {}),
            "input": {
                "messages": deepcopy(messages),
                "prompt_token_ids": tokens,
                "prompt_tokens_sha256": digest(tokens),
                "chat_template": self.template,
                "chat_template_sha256": digest(self.template) if self.template else None,
            },
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(output.token_ids),
                "total_tokens": len(tokens) + len(output.token_ids),
            },
            "raw": raw,
        }
        canonical_json(response)
        return response


def vllm_provider(*, config):
    """Execute-only factory for ``--provider-factory warbias_kit.backends:vllm_provider``."""
    checked = validate_vllm_config(config)
    runtime = importlib.import_module("vllm")
    return VllmProvider(checked, runtime)
