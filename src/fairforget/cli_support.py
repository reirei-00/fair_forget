"""Shared CLI file loading and explicit execute-only factory binding."""

import importlib
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_factory(value):
    module, separator, name = value.partition(":")
    if not separator or not module or not name:
        raise ValueError("A factory must be written as module:callable")
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise ValueError("The configured factory is not callable")
    return result


def execution_factory(args):
    if not args.execute:
        return None
    if not args.provider_factory:
        raise ValueError("--execute requires an explicit --provider-factory module:callable")
    return load_factory(args.provider_factory)


def provider_from_args(args, config):
    factory = execution_factory(args)
    return factory(config=config) if factory else None


def add_execution_options(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--provider-factory", help="Factory(config=...) returning a generation provider"
    )


def add_generation_options(parser):
    parser.add_argument("--config", type=Path, required=True)
    add_execution_options(parser)
