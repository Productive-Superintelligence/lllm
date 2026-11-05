"""LLLM shell command line interface."""

from __future__ import annotations

import argparse
import ipaddress
import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from .. import __version__
from .loader import load_strategy
from .server import create_strategy_app

LOG_LEVELS = {"critical", "debug", "error", "info", "trace", "warning"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lllm shell", description="LLLM shell strategy tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subcommands.add_parser("inspect", help="Inspect a strategy")
    inspect_cmd.add_argument("target", help="strategy.py, directory, or module:factory")
    inspect_cmd.add_argument("--json", action="store_true")

    serve_cmd = subcommands.add_parser("serve", help="Serve a strategy as FastAPI")
    serve_cmd.add_argument("target", help="strategy.py, directory, or module:factory")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8400)
    serve_cmd.add_argument("--log-level", default="info")

    launch_cmd = subcommands.add_parser("launch", help="Alias for serve")
    launch_cmd.add_argument("target", help="strategy.py, directory, or module:factory")
    launch_cmd.add_argument("--host", default="127.0.0.1")
    launch_cmd.add_argument("--port", type=int, default=8400)
    launch_cmd.add_argument("--log-level", default="info")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        strategy = _load_or_error(parser, args.target)
        payload = jsonable_encoder(strategy.info())
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{payload['name']}: {payload.get('description', '')}")
            for resource in payload.get("resources", []):
                ref = resource.get("ref") or resource.get("entrypoint") or ""
                print(f"{resource['kind']} {resource['name']} {ref}".rstrip())
        return 0

    if args.command in {"serve", "launch"}:
        try:
            host = _serve_host(args.host)
            port = _serve_port(args.port)
            log_level = _serve_log_level(args.log_level)
        except ValueError as exc:
            parser.error(str(exc))
        strategy = _load_or_error(parser, args.target)
        import uvicorn

        uvicorn.run(
            create_strategy_app(strategy),
            host=host,
            port=port,
            log_level=log_level,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _load_or_error(parser: argparse.ArgumentParser, target: str) -> Any:
    try:
        return load_strategy(target)
    except (OSError, TypeError, ValueError, AttributeError, ImportError) as exc:
        parser.error(str(exc))
        raise


def _serve_host(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("serve host must be a non-empty string")
    host = value
    if any(ch.isspace() for ch in host) or "/" in host or "\\" in host:
        raise ValueError("serve host must be a host name or address, not a URL or path")
    if ":" in host:
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("serve host must be a host name or address, not host:port") from exc
    return host


def _serve_port(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not (1 <= value <= 65535)
    ):
        raise ValueError("serve port must be an integer between 1 and 65535")
    return value


def _serve_log_level(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ch.isspace() for ch in value)
    ):
        raise ValueError(_log_level_error())
    log_level = value.lower()
    if log_level not in LOG_LEVELS:
        raise ValueError(_log_level_error())
    return log_level


def _log_level_error() -> str:
    return "serve log level must be one of: critical, debug, error, info, trace, warning"
