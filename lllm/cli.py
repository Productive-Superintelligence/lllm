"""LLLM command line tools.

``lllm create|init|setup|inspect|serve`` cover tactics and packages, and
``lllm hub`` dispatches to the hub CLI. Other packages add commands through
the ``lllm.commands`` entry-point group, which maps a command name to a module
with ``main(argv)``: installing sssn adds ``lllm network`` and installing aaax
adds ``lllm shell``.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import importlib.util
import ipaddress
import json
import os
import sys
import tempfile
from importlib.metadata import entry_points
from typing import Any, Callable, NamedTuple

from .create import create_project
from .hub import tactic_resource
from .protocol import Tactic
from .services import create_tactic_app

LOG_LEVELS = {"critical", "debug", "error", "info", "trace", "warning"}
LOG_LEVEL_ERROR = (
    "serve log level must be one of: critical, debug, error, info, trace, warning"
)
DEFAULT_SERVE_PORT = 8000
MODULE_COMMANDS = {
    "hub": ("lllm.hub.cli", "Validate, publish, and index packages"),
}
CORE_COMMANDS = frozenset(
    {"create", "init", "setup", "inspect", "serve", *MODULE_COMMANDS}
)
#: Core commands that import user code (``module:attr`` entrypoints and package
#: entries); only these put the working directory on ``sys.path``.
USER_CODE_COMMANDS = frozenset({"inspect", "serve", "hub"})
PLUGIN_GROUP = "lllm.commands"
#: Plugin commands lllm knows by name, used when the package is installed
#: without its entry point: command -> (package, CLI module, help).
PLUGIN_FALLBACKS = {
    "network": ("sssn", "sssn.cli", "SSSN channel stores"),
    "shell": ("aaax", "aaax.cli", "AAAX strategy shell"),
}


class _PluginCommand(NamedTuple):
    """A command another package adds to ``lllm``."""

    package: str
    module: str
    help: str
    installed: bool
    load: Callable[[], Any]


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    command = raw_args[0] if raw_args else None
    # Plugin commands load and run before the working directory joins
    # sys.path, so a local aaax.py or sssn.py cannot stand in for the
    # installed package.
    plugins = {} if command in CORE_COMMANDS else _plugin_commands()
    if command in plugins:
        return _run_plugin(command, plugins[command], raw_args[1:])
    if command in USER_CODE_COMMANDS:
        _import_from_working_directory()
    if command in MODULE_COMMANDS:
        module = importlib.import_module(MODULE_COMMANDS[command][0])
        return module.main(raw_args[1:])

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="lllm",
        description="Build, package, and serve agentic tactics.",
    )
    parser.add_argument("--version", action="version", version=f"lllm {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create_cmd = subcommands.add_parser(
        "create",
        help="Create a project (native templates or a typed service scaffold)",
        description=(
            "lllm create NAME [--template minimal|pipeline|proxy|research|service|"
            "plain|pydantic-ai|native|PATH]. The older form `lllm create TEMPLATE "
            "NAME` also works."
        ),
    )
    create_cmd.add_argument("first", metavar="NAME", help="Project name")
    create_cmd.add_argument("second", nargs="?", help=argparse.SUPPRESS)
    create_cmd.add_argument(
        "--template",
        default=None,
        help="Built-in template or a folder with lllm-template.toml (default: minimal)",
    )
    create_cmd.add_argument("--model", default="gpt-4o-mini", help="Default model in configs")
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
        help=(
            "Where keys are stored: auto (the default) picks a usable OS keyring "
            "when there is one, and the env file otherwise"
        ),
    )
    setup_cmd.add_argument(
        "--env-file",
        default=".env.local",
        help=(
            "Env file to read API keys from; keys are stored here with "
            "--credentials env, or with auto when no usable keyring is available"
        ),
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
        help="Show tactic metadata, or what serve would launch for an app or package",
    )
    inspect_cmd.add_argument("target", help="module:attribute entrypoint or package path")
    inspect_cmd.add_argument("--json", action="store_true", help="Emit JSON")
    _add_package_options(inspect_cmd, serving=False)

    serve_cmd = subcommands.add_parser(
        "serve",
        help="Serve a tactic or app entrypoint, or one package resource",
    )
    serve_cmd.add_argument("target", help="module:attribute entrypoint or package path")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument(
        "--port",
        type=int,
        help=f"Port (default: the resource's declared port, else {DEFAULT_SERVE_PORT})",
    )
    serve_cmd.add_argument("--log-level", default="info")
    _add_package_options(serve_cmd, serving=True)
    serve_cmd.add_argument(
        "--credentials",
        choices=("auto", "keyring", "env"),
        default="auto",
        help=(
            "Where interactive setup stores the keys you type: auto (the default) "
            "picks the OS keyring when a usable one is available and the env file "
            "otherwise; --no-keyring makes it the env file, and refuses "
            "--credentials keyring"
        ),
    )
    serve_cmd.add_argument(
        "--skip-key-check",
        action="store_true",
        help="Start without checking the package's declared API keys",
    )

    for name, (_module, help_text) in MODULE_COMMANDS.items():
        subcommands.add_parser(name, help=help_text, add_help=False)
    not_installed = []
    for name, plugin in sorted(plugins.items()):
        if plugin.installed:
            subcommands.add_parser(name, help=plugin.help, add_help=False)
        else:
            not_installed.append(f"lllm {name} (pip install {plugin.package})")
    if not_installed:
        parser.epilog = f"More commands: {', '.join(not_installed)}."

    args = parser.parse_args(raw_args)

    if args.command == "create":
        if args.second is not None:  # legacy: lllm create TEMPLATE NAME
            template, name = args.first, args.second
        else:
            template, name = args.template or "minimal", args.first
        try:
            result = create_project(
                template,
                name,
                directory=args.directory,
                force=args.force,
                model=args.model,
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
        if _is_package_target(parser, args.target):
            return _inspect_package(parser, args)
        target = _load_cli_entrypoint(parser, args.target)
        if not isinstance(target, Tactic):
            return _inspect_app(args, target)
        resource = tactic_resource(target)
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

        if _is_package_target(parser, args.target):
            if not args.skip_key_check:
                _ensure_launch_keys(parser, args)
            launched = _load_launch_app(parser, args)
            app, port = launched.app, port or launched.port
        else:
            app = _load_cli_entrypoint(parser, args.target)
            if isinstance(app, Tactic):
                app = create_tactic_app(app)
            port = port or DEFAULT_SERVE_PORT

        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _plugin_commands() -> dict[str, _PluginCommand]:
    """Commands other packages add to ``lllm``, by name.

    Entry points in the ``lllm.commands`` group name a module with
    ``main(argv)``. The commands in :data:`PLUGIN_FALLBACKS` are listed even
    when their package is missing, so ``lllm network`` can say what to install.
    """

    plugins = {
        name: _PluginCommand(
            package=package,
            module=module,
            help=help_text,
            installed=_importable(package),
            load=functools.partial(importlib.import_module, module),
        )
        for name, (package, module, help_text) in PLUGIN_FALLBACKS.items()
    }
    found: set[str] = set()
    for entry_point in entry_points(group=PLUGIN_GROUP):
        name = entry_point.name
        if name in CORE_COMMANDS or name in found:
            continue
        found.add(name)
        dist = getattr(entry_point, "dist", None)
        package = dist.name if dist is not None else entry_point.module.split(".")[0]
        fallback = PLUGIN_FALLBACKS.get(name)
        plugins[name] = _PluginCommand(
            package=package,
            module=entry_point.module,
            help=fallback[2] if fallback else f"Added by {package}",
            installed=True,
            load=entry_point.load,
        )
    return plugins


def _run_plugin(name: str, plugin: _PluginCommand, argv: list[str]) -> int:
    try:
        target = plugin.load()
    except ModuleNotFoundError as exc:
        # Only the plugin package itself counts as missing; an installed plugin
        # that cannot import one of its own dependencies raises as usual.
        if exc.name is None or not f"{plugin.module}.".startswith(f"{exc.name}."):
            raise
        print(
            f"lllm {name} needs the {plugin.package} package: pip install {plugin.package}",
            file=sys.stderr,
        )
        return 2
    command = getattr(target, "main", target)
    if not callable(command):
        raise TypeError(f"lllm {name}: {plugin.module} has no main(argv)")
    return command(argv)


def _importable(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _import_from_working_directory() -> None:
    """Let ``module:attr`` entrypoints import modules from the current directory.

    ``python -m`` and Uvicorn do this; a console script does not.
    """

    cwd = os.getcwd()
    if "" not in sys.path and cwd not in sys.path:
        sys.path.insert(0, cwd)


def _add_package_options(
    command: argparse.ArgumentParser,
    *,
    serving: bool,
) -> None:
    """The package options of ``inspect`` and ``serve``.

    Only serve opens a channel store or stores the keys it asks for, so
    inspect hides ``--store`` (it still accepts it) and its key options
    only read.
    """

    command.add_argument("--resource", help="Package resource, e.g. services.api")
    command.add_argument(
        "--store",
        help="Channel store root for channel resources" if serving else argparse.SUPPRESS,
    )
    command.add_argument(
        "--env-file",
        default=".env.local",
        help=(
            "Env file to read API keys from; typed keys are stored here with "
            "--credentials env, with --no-keyring, or with auto when no usable "
            "keyring is available"
            if serving
            else "Env file to read API keys from"
        ),
    )
    command.add_argument(
        "--no-keyring",
        action="store_true",
        help=(
            "Neither read API keys from the OS keyring nor store typed keys there; "
            "they go to --env-file"
            if serving
            else "Do not read API keys from the OS keyring"
        ),
    )


def _is_package_target(parser: argparse.ArgumentParser, target: str) -> bool:
    from .launch import LaunchError, _target_path

    try:
        path = _target_path(target)
    except LaunchError as exc:
        parser.error(str(exc))
        raise
    if path.name == "lllm.toml":
        return True
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:  # e.g. EACCES, or a name that is too long
        if _has_entrypoint_form(target):
            # Not a path, so a working folder that cannot be searched
            # does not stop it.
            return False
        parser.error(f"cannot read {target}: {exc.strerror or exc}")
        raise
    return True


def _inspect_package(parser: argparse.ArgumentParser, args: Any) -> int:
    requirements, resolution = _resolve_requirements(
        parser,
        args.target,
        resource=args.resource,
        env_file=args.env_file,
        use_keyring=not args.no_keyring,
    )
    # Building a channel app opens its store, which creates the store folder
    # and the declared channels; inspecting builds it over a throwaway store.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as store:
        launched = _load_launch_app(parser, args, store=store)
        routes = _route_paths(launched.app)
    payload = {
        "label": launched.label,
        "kind": launched.kind,
        "port": launched.port,
        "routes": routes,
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


def _inspect_app(args: Any, app: Any) -> int:
    payload = {
        "label": args.target,
        "kind": "app",
        "app": type(app).__name__,
        "port": DEFAULT_SERVE_PORT,
        "routes": _route_paths(app),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"{payload['label']} ({payload['app']} app)")
    print(f"port: {payload['port']}")
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
    """Check a package's API keys before serving it, asking for missing ones."""

    from .credentials import (
        CredentialError,
        configure_missing_api_keys,
        format_missing_requirements,
        resolve_api_keys,
    )

    if args.no_keyring and args.credentials == "keyring":
        parser.error(
            "--no-keyring does not read the OS keyring, so --credentials keyring "
            "would store keys it cannot see; use --credentials env"
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
                # The check below reads the env file but, with --no-keyring,
                # not the keyring, so keys typed here go to the env file.
                credentials="env" if args.no_keyring else args.credentials,
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


def _load_launch_app(
    parser: argparse.ArgumentParser,
    args: Any,
    *,
    store: str | None = None,
) -> Any:
    from .launch import LaunchError, load_launch_app

    try:
        return load_launch_app(
            args.target,
            resource=args.resource,
            store=args.store if store is None else store,
        )
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
    ``CallableTactic`` unless they already return a ``Tactic``. A FastAPI or
    Starlette app, or a factory that returns one, raises ``TypeError``.
    """

    target = _load_entrypoint(entrypoint)
    if not isinstance(target, Tactic):
        raise TypeError(
            f"Entrypoint {entrypoint} resolves to a FastAPI or Starlette app, not a tactic"
        )
    return target


def _load_entrypoint(entrypoint: str) -> Any:
    """Load ``module:attribute`` as a FastAPI or Starlette app, or a tactic.

    An app, or a factory that returns one, is returned as it is; anything
    else becomes a tactic as :func:`coerce_tactic` makes one. A factory is
    called once.
    """

    from .launch import _entrypoint_target

    if (
        not isinstance(entrypoint, str)
        or not entrypoint
        or entrypoint != entrypoint.strip()
        or not _has_entrypoint_form(entrypoint)
    ):
        raise ValueError("Entrypoint must have the form 'module:attribute'.")
    module_name, _, attr_path = entrypoint.partition(":")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    try:
        return _entrypoint_target(value)
    except TypeError as exc:
        raise TypeError(
            f"Entrypoint {entrypoint} did not resolve to a tactic, callable, or "
            f"app: {exc}"
        ) from exc


def _load_cli_entrypoint(parser: argparse.ArgumentParser, entrypoint: str) -> Any:
    try:
        return _load_entrypoint(entrypoint)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        raise


def _has_entrypoint_form(value: str) -> bool:
    """Whether ``value`` is ``module:attribute``, as dotted names."""

    module_name, sep, attr_path = value.partition(":")
    return bool(sep) and _entrypoint_segments(module_name) and _entrypoint_segments(attr_path)


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


if __name__ == "__main__":
    raise SystemExit(main())
