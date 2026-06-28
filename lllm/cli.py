"""LLLM command line tools."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from typing import Any

from .integrations import tactic_resource
from .protocol import Tactic
from .runtimes import as_tactic
from .services import create_tactic_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lllm")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subcommands.add_parser("inspect", help="Show tactic metadata")
    inspect_cmd.add_argument("entrypoint", help="Python entrypoint, e.g. demo:build_tactic")
    inspect_cmd.add_argument("--json", action="store_true", help="Emit JSON")

    serve_cmd = subcommands.add_parser("serve", help="Serve one tactic entrypoint")
    serve_cmd.add_argument("entrypoint", help="Python entrypoint, e.g. demo:build_tactic")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--log-level", default="info")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        tactic = load_tactic_entrypoint(args.entrypoint)
        resource = tactic_resource(tactic)
        if args.json:
            print(json.dumps(resource, indent=2, sort_keys=True))
        else:
            print(f"{resource['name']} ({resource['runtime']})")
            print(resource["description"])
            print("capabilities:", ", ".join(resource["capabilities"]))
        return 0

    if args.command == "serve":
        import uvicorn

        tactic = load_tactic_entrypoint(args.entrypoint)
        app = create_tactic_app(tactic)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def load_tactic_entrypoint(entrypoint: str) -> Tactic[Any, Any]:
    """Load a tactic from ``module:attribute``.

    Factories are called with no arguments. Plain callables are wrapped as
    ``CallableTactic`` unless they already return a ``Tactic``.
    """

    module_name, sep, attr_path = entrypoint.partition(":")
    if not sep or not module_name or not attr_path:
        raise ValueError("Entrypoint must have the form 'module:attribute'.")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    if isinstance(value, type) and issubclass(value, Tactic):
        return value()
    if isinstance(value, Tactic):
        return value
    if callable(value):
        if not _can_call_without_args(value):
            return as_tactic(value)
        produced = value()
        if isinstance(produced, type) and issubclass(produced, Tactic):
            return produced()
        if isinstance(produced, Tactic):
            return produced
        if callable(produced):
            return as_tactic(produced)
        return as_tactic(value)
    raise TypeError(f"Entrypoint did not resolve to a tactic or callable: {entrypoint}")


def _can_call_without_args(value: Any) -> bool:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.default is inspect.Parameter.empty and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
