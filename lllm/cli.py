"""LLLM command line tools.

``lllm create|init|setup|inspect|serve`` cover tactics and packages.
``lllm hub``, ``lllm network`` (SSSN), and ``lllm shell`` (AAAX) dispatch to
their module CLIs.
"""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from .create import create_project
from .hub import tactic_resource
from .protocol import Tactic
from .runtimes import as_tactic
from .services import create_tactic_app

LOG_LEVELS = {"critical", "debug", "error", "info", "trace", "warning"}
LOG_LEVEL_ERROR = (
    "serve log level must be one of: critical, debug, error, info, trace, warning"
)
DEFAULT_SERVE_PORT = 8000
MODULE_COMMANDS = {
    "hub": ("lllm.hub.cli", "Validate, publish, and index packages"),
    "network": ("lllm.network.cli", "SSSN channel stores (experimental)"),
    "shell": ("lllm.shell.cli", "AAAX strategy shell (experimental)"),
}


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in MODULE_COMMANDS:
        module = importlib.import_module(MODULE_COMMANDS[raw_args[0]][0])
        return module.main(raw_args[1:])

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="lllm",
        description="Build, package, and serve agentic tactics.",
    )
    parser.add_argument("--version", action="version", version=f"lllm {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create_cmd = subcommands.add_parser("create", help="Create a tactic service project")
    create_cmd.add_argument(
        "template",
        choices=["plain", "pydantic-ai", "native"],
        help="Project template",
    )
    create_cmd.add_argument("name", help="Project name")
    create_cmd.add_argument("--directory", default=".", help="Parent directory")
    create_cmd.add_argument("--force", action="store_true", help="Overwrite existing files")

    init_cmd = subcommands.add_parser("init", help="Write an lllm.toml package manifest")
    init_cmd.add_argument("path", nargs="?", default=".", help="Package folder")
    init_cmd.add_argument("--name", help="Package name (defaults to the folder name)")
    init_cmd.add_argument("--org", default="local")
    init_cmd.add_argument("--kind", default="mixed")
    init_cmd.add_argument("--force", action="store_true", help="Overwrite lllm.toml")

    setup_cmd = subcommands.add_parser(
        "setup",
        help="Set up the API keys a package declares",
    )
    setup_cmd.add_argument(
        "target",
        nargs="?",
        help="Package path, lllm.toml path, or module:attribute entrypoint",
    )
    setup_cmd.add_argument("--resource", help="Set up one resource, e.g. services.api")
    setup_cmd.add_argument(
        "--credentials",
        choices=("auto", "keyring", "env"),
        default="auto",
        help="Store entered keys in the OS keyring or a local env file",
    )
    setup_cmd.add_argument(
        "--env-file",
        default=".env.local",
        help="Local env file used when --credentials env is selected",
    )
    setup_cmd.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=value",
        help="Provide a key non-interactively",
    )
    setup_cmd.add_argument(
        "--show-conventions",
        action="store_true",
        help="Print common provider API key names",
    )

    inspect_cmd = subcommands.add_parser(
        "inspect",
        help="Show tactic metadata, or what serve would launch for a package",
    )
    inspect_cmd.add_argument("target", help="module:attribute entrypoint or package path")
    inspect_cmd.add_argument("--json", action="store_true", help="Emit JSON")
    _add_package_options(inspect_cmd)

    serve_cmd = subcommands.add_parser(
        "serve",
        help="Serve a tactic entrypoint or one package resource over FastAPI",
    )
    serve_cmd.add_argument("target", help="module:attribute entrypoint or package path")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument(
        "--port",
        type=int,
        help=f"Port (default: the resource's declared port, else {DEFAULT_SERVE_PORT})",
    )
    serve_cmd.add_argument("--log-level", default="info")
    _add_package_options(serve_cmd)
    serve_cmd.add_argument(
        "--credentials",
        choices=("auto", "keyring", "env"),
        default="auto",
        help="Where interactive setup should store missing keys",
    )
    serve_cmd.add_argument(
        "--skip-key-check",
        action="store_true",
        help="Start without checking the package's declared API keys",
    )

    for name, (_module, help_text) in MODULE_COMMANDS.items():
        subcommands.add_parser(name, help=help_text, add_help=False)

    args = parser.parse_args(raw_args)

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

    if args.command == "init":
        from .hub import init_package

        try:
            target = init_package(
                args.path,
                name=args.name,
                org=args.org,
                kind=args.kind,
                force=args.force,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(target)
        return 0

    if args.command == "setup":
        return _setup_command(parser, args)

    if args.command == "inspect":
        if _is_package_target(args.target):
            return _inspect_package(parser, args)
        tactic = _load_cli_tactic(parser, args.target)
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
            port = _serve_port(args.port) if args.port is not None else None
            log_level = _serve_log_level(args.log_level)
        except ValueError as exc:
            parser.error(str(exc))

        if _is_package_target(args.target):
            if not args.skip_key_check:
                _ensure_launch_keys(parser, args)
            launched = _load_launch_app(parser, args)
            app, port = launched.app, port or launched.port
        else:
            app = create_tactic_app(_load_cli_tactic(parser, args.target))
            port = port or DEFAULT_SERVE_PORT

        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_package_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--resource", help="Package resource, e.g. services.api")
    command.add_argument("--store", help="Channel store root for channel resources")
    command.add_argument(
        "--env-file",
        default=".env.local",
        help="Local env file to read for required API keys",
    )
    command.add_argument(
        "--no-keyring",
        action="store_true",
        help="Do not read API keys from the OS keyring",
    )


def _is_package_target(target: str) -> bool:
    path = Path(target).expanduser()
    return path.exists() or path.name == "lllm.toml"


def _inspect_package(parser: argparse.ArgumentParser, args: Any) -> int:
    requirements, resolution = _resolve_requirements(
        parser,
        args.target,
        resource=args.resource,
        env_file=args.env_file,
        use_keyring=not args.no_keyring,
    )
    launched = _load_launch_app(parser, args)
    payload = {
        "label": launched.label,
        "kind": launched.kind,
        "port": launched.port,
        "routes": _route_paths(launched.app),
        "api_keys": resolution.public_payload(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"{payload['label']} ({payload['kind']})")
    print(f"port: {payload['port']}")
    if requirements:
        print("api keys:")
        for status in resolution.statuses:
            marker = "ready" if status.ready else "missing"
            description = status.requirement.description
            detail = f" - {description}" if description else ""
            print(f"  {status.requirement.name}: {marker}{detail}")
    for route in payload["routes"]:
        print(route)
    return 0


def _setup_command(parser: argparse.ArgumentParser, args: Any) -> int:
    from .credentials import (
        CredentialError,
        configure_missing_api_keys,
        conventional_requirements,
        format_missing_requirements,
        parse_key_values,
    )
    from .launch import LaunchError

    try:
        provided = parse_key_values(args.set)
        if args.show_conventions:
            for requirement in conventional_requirements():
                print(f"{requirement.name}: {requirement.description}")
            if args.target is None:
                return 0
        if args.target is None:
            print("Run `lllm setup PACKAGE` to configure keys declared by a package.")
            print("Use `lllm setup --show-conventions` to list common provider names.")
            return 0
        requirements, resolution = _resolve_requirements(
            parser,
            args.target,
            resource=args.resource,
            env_file=args.env_file,
            use_keyring=True,
        )
        if not requirements:
            print("No API key requirements declared for this target.")
            return 0
        declared = {requirement.name: requirement for requirement in requirements}
        for key in provided:
            if key not in declared:
                print(f"warning: {key} is not declared by this package")
        missing = list(resolution.missing)
        provided_declared = {
            key: value for key, value in provided.items() if key in declared
        }
        still_prompt = [
            requirement for requirement in missing if requirement.name not in provided
        ]
        if not missing and not provided:
            print("All required API keys are already available.")
            return 0
        if still_prompt and not sys.stdin.isatty():
            parser.error(
                "missing required API key(s):\n"
                f"{format_missing_requirements(still_prompt)}\n\n"
                "Use --set NAME=value for non-interactive setup."
            )
        prompted = {requirement.name for requirement in still_prompt}
        to_configure = [
            *still_prompt,
            *[declared[key] for key in provided_declared if key not in prompted],
        ]
        backend = configure_missing_api_keys(
            to_configure,
            credentials=args.credentials,
            env_file=args.env_file,
            provided=provided_declared,
        )
        print(f"Configured {len(to_configure)} API key(s) via {backend}.")
        return 0
    except (CredentialError, ImportError, LaunchError, OSError, ValueError) as exc:
        parser.error(str(exc))
        raise


def _ensure_launch_keys(parser: argparse.ArgumentParser, args: Any) -> None:
    from .credentials import (
        CredentialError,
        configure_missing_api_keys,
        format_missing_requirements,
        resolve_api_keys,
    )

    requirements, resolution = _resolve_requirements(
        parser,
        args.target,
        resource=args.resource,
        env_file=args.env_file,
        use_keyring=not args.no_keyring,
    )
    if not requirements or resolution.ok:
        return
    if sys.stdin.isatty():
        try:
            backend = configure_missing_api_keys(
                resolution.missing,
                credentials=args.credentials,
                env_file=args.env_file,
            )
            print(f"Configured {len(resolution.missing)} API key(s) via {backend}.")
            resolution = resolve_api_keys(
                requirements,
                env_file=args.env_file,
                use_keyring=not args.no_keyring,
            )
        except CredentialError as exc:
            parser.error(str(exc))
    if not resolution.ok:
        parser.error(
            "missing required API key(s):\n"
            f"{format_missing_requirements(resolution.missing)}\n\n"
            f"Run `lllm setup {args.target}` or set the missing names in your environment."
        )


def _resolve_requirements(
    parser: argparse.ArgumentParser,
    target: str,
    *,
    resource: str | None,
    env_file: str,
    use_keyring: bool,
) -> tuple[Any, Any]:
    from .credentials import CredentialError, resolve_api_keys
    from .launch import LaunchError, load_launch_api_key_requirements

    try:
        requirements = load_launch_api_key_requirements(target, resource=resource)
        resolution = resolve_api_keys(
            requirements,
            env_file=env_file,
            use_keyring=use_keyring,
        )
    except (CredentialError, ImportError, LaunchError, OSError, ValueError) as exc:
        parser.error(str(exc))
        raise
    return requirements, resolution


def _load_launch_app(parser: argparse.ArgumentParser, args: Any) -> Any:
    from .launch import LaunchError, load_launch_app

    try:
        return load_launch_app(args.target, resource=args.resource, store=args.store)
    except (ImportError, LaunchError, OSError, ValueError, AttributeError, TypeError) as exc:
        parser.error(str(exc))
        raise


def _route_paths(app: Any) -> list[str]:
    routes = []
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path:
            prefix = ",".join(sorted(methods or []))
            routes.append(f"{prefix} {path}" if prefix else str(path))
    return sorted(routes)


def load_tactic_entrypoint(entrypoint: str) -> Tactic[Any, Any]:
    """Load a tactic from ``module:attribute``.

    Factories are called with no arguments. Plain callables are wrapped as
    ``CallableTactic`` unless they already return a ``Tactic``.
    """

    if (
        not isinstance(entrypoint, str)
        or not entrypoint
        or entrypoint != entrypoint.strip()
    ):
        raise ValueError("Entrypoint must have the form 'module:attribute'.")
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
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        raise


def _entrypoint_segments(value: str) -> bool:
    return all(
        part
        and not any(ch.isspace() for ch in part)
        and not any(ch in part for ch in "/\\:%")
        for part in value.split(".")
    )


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
            raise ValueError(
                "serve host must be a host name or address, not host:port"
            ) from exc
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
        raise ValueError(LOG_LEVEL_ERROR)
    log_level = value.lower()
    if log_level not in LOG_LEVELS:
        raise ValueError(LOG_LEVEL_ERROR)
    return log_level


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
