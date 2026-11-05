"""Local LLLM hub command line tools."""

from __future__ import annotations

import argparse
import ipaddress
import sys

from .local import LocalHub, PublishValidationError
from .manifest import init_package
from .validator import validate_package

LOG_LEVELS = {"critical", "debug", "error", "info", "trace", "warning"}
LOG_LEVEL_ERROR = (
    "serve log level must be one of: critical, debug, error, info, trace, warning"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lllm hub", description="Local LLLM package hub")
    parser.add_argument("--hub", default=".lllm/hub", help="Local hub directory")
    subcommands = parser.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser("init", help="Prepare package metadata")
    init.add_argument("path", nargs="?", default=".")
    init.add_argument("--name")
    init.add_argument("--org", default="local")
    init.add_argument("--kind", default="mixed")
    init.add_argument("--force", action="store_true")

    validate = subcommands.add_parser("validate", help="Validate a package")
    validate.add_argument("path")

    publish = subcommands.add_parser("publish", help="Publish a package")
    publish.add_argument("path")
    publish.add_argument("--local", action="store_true", help="Publish to the local hub")
    publish.add_argument("--no-validate", action="store_true")

    get = subcommands.add_parser("get", help="Download a package from the local hub")
    get.add_argument("identifier")
    get.add_argument("--dest", default=".")
    get.add_argument("--version")

    subcommands.add_parser("list", help="List local packages")

    card = subcommands.add_parser("card", help="Render a package card")
    card.add_argument("identifier")
    card.add_argument("--version")

    agent_card = subcommands.add_parser("agent-card", help="Render a package agent card")
    agent_card.add_argument("identifier")
    agent_card.add_argument("--version")

    config = subcommands.add_parser("config-template", help="Render .lllm/config.toml")
    config.add_argument("identifier")
    config.add_argument("--version")

    index = subcommands.add_parser(
        "index",
        help="Write a static catalog (data/catalog.json + zips) for the hub web app",
    )
    index.add_argument("output", help="Output folder, e.g. site/hub")
    index.add_argument("--generated-at", help="Fixed timestamp for reproducible builds")

    serve = subcommands.add_parser("serve", help="Serve the local package hub API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--log-level", default="info")

    args = parser.parse_args(argv)
    serve_host = None
    serve_port = None
    serve_log_level = None
    if args.command == "serve":
        try:
            serve_host = _serve_host(args.host)
            serve_port = _serve_port(args.port)
            serve_log_level = _serve_log_level(args.log_level)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "init":
        try:
            path = init_package(
                args.path,
                name=args.name,
                org=args.org,
                kind=args.kind,
                force=args.force,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(path)
        return 0

    if args.command == "validate":
        report = validate_package(args.path)
        for issue in report.issues:
            print(f"{issue.level}: {issue.code}: {issue.message}", file=sys.stderr)
        print("ok" if report.ok else "failed")
        return 0 if report.ok else 1

    try:
        hub = LocalHub(args.hub)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "publish":
        if not args.local:
            parser.error("Only local publish is supported in this phase. Use --local.")
        try:
            record = hub.publish(args.path, validate=not args.no_validate)
        except PublishValidationError as exc:
            for issue in exc.report.issues:
                print(f"{issue.level}: {issue.code}: {issue.message}", file=sys.stderr)
            print("failed")
            return 1
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(record.key)
        return 0 if record.validation.ok or args.no_validate else 1

    if args.command == "get":
        try:
            print(hub.download(args.identifier, args.dest, version=args.version))
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "list":
        for record in hub.list():
            print(record.key)
        return 0

    if args.command == "index":
        from .catalog import build_catalog

        catalog = build_catalog(hub, args.output, generated_at=args.generated_at)
        print(f"indexed {len(catalog['packages'])} package(s) into {args.output}")
        return 0

    if args.command == "card":
        try:
            print(hub.card(args.identifier, version=args.version), end="")
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "agent-card":
        try:
            print(hub.agent_card(args.identifier, version=args.version), end="")
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "config-template":
        try:
            print(hub.config_template(args.identifier, version=args.version), end="")
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "serve":
        from .server import create_app

        import uvicorn

        uvicorn.run(
            create_app(hub=hub),
            host=serve_host,
            port=serve_port,
            log_level=serve_log_level,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


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
