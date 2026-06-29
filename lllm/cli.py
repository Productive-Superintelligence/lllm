"""LLLM command line tools."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from typing import Any

from .create import create_project
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

    create_cmd = subcommands.add_parser("create", help="Create a tactic service project")
    create_cmd.add_argument(
        "template",
        choices=["plain", "pydantic-ai", "native"],
        help="Project template",
    )
    create_cmd.add_argument("name", help="Project name")
    create_cmd.add_argument("--directory", default=".", help="Parent directory")
    create_cmd.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        tactic = _load_cli_tactic(parser, args.entrypoint)
        resource = tactic_resource(tactic)
        if args.json:
            print(json.dumps(resource, indent=2, sort_keys=True))
        else:
            print(f"{resource['name']} ({resource['runtime']})")
            print(resource["description"])
            print("capabilities:", ", ".join(resource["capabilities"]))
        return 0

    if args.command == "serve":
        try:
            host = _serve_host(args.host)
            port = _serve_port(args.port)
        except ValueError as exc:
            parser.error(str(exc))
        import uvicorn

        tactic = _load_cli_tactic(parser, args.entrypoint)
        app = create_tactic_app(tactic)
        uvicorn.run(app, host=host, port=port, log_level=args.log_level)
        return 0

    if args.command == "create":
        try:
            result = create_project(
                args.template,
                args.name,
                directory=args.directory,
                force=args.force,
            )
        except (FileExistsError, ValueError) as exc:
            parser.error(str(exc))
        print(f"created {result.template} project at {result.path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def load_tactic_entrypoint(entrypoint: str) -> Tactic[Any, Any]:
    """Load a tactic from ``module:attribute``.

    Factories are called with no arguments. Plain callables are wrapped as
    ``CallableTactic`` unless they already return a ``Tactic``.
    """

    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError("Entrypoint must have the form 'module:attribute'.")
    entrypoint = entrypoint.strip()
    module_name, sep, attr_path = entrypoint.partition(":")
    if (
        not sep
        or not _entrypoint_segments(module_name)
        or not _entrypoint_segments(attr_path)
    ):
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


def _load_cli_tactic(parser: argparse.ArgumentParser, entrypoint: str) -> Tactic[Any, Any]:
    try:
        return load_tactic_entrypoint(entrypoint)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
        raise


def _entrypoint_segments(value: str) -> bool:
    return all(part and not any(ch.isspace() for ch in part) for part in value.split("."))


def _serve_host(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("serve host must be a non-empty string")
    host = value.strip()
    if any(ch.isspace() for ch in host) or "/" in host or "\\" in host:
        raise ValueError("serve host must be a host name or address, not a URL or path")
    return host


def _serve_port(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (1 <= value <= 65535)
    ):
        raise ValueError("serve port must be an integer between 1 and 65535")
    return value


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
