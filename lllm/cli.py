from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.templates import (
    DEFAULT_TEMPLATE,
    list_builtin_template_names,
    normalize_package_name,
    normalize_pyproject_name,
    render_template,
    resolve_template,
    title_from_name,
)

DEFAULT_MODEL = "gpt-4o-mini"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lllm", description="LLLM helper CLI.")
    subparsers = parser.add_subparsers(dest="command")

    # ── create ──────────────────────────────────────────────────────────────
    create_parser = subparsers.add_parser(
        "create", help="Create a new LLLM project scaffold."
    )
    create_parser.add_argument("project_name", help="Project folder to create.")
    create_parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help=(
            "Built-in template name or local template folder "
            f"(default: {DEFAULT_TEMPLATE}; built-ins: "
            f"{', '.join(list_builtin_template_names())})."
        ),
    )
    create_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Default model to put in configs/default.yaml (default: {DEFAULT_MODEL}).",
    )

    # ── pkg install ──────────────────────────────────────────────────────────
    pkg_parser = subparsers.add_parser("pkg", help="Package management commands.")
    pkg_sub = pkg_parser.add_subparsers(dest="pkg_command")

    install_parser = pkg_sub.add_parser(
        "install", help="Install a package from a .zip file."
    )
    install_parser.add_argument("zip_path", help="Path to the package .zip file.")
    install_parser.add_argument(
        "--alias",
        default=None,
        help="Install under this name instead of the name declared in lllm.toml.",
    )
    install_parser.add_argument(
        "--scope",
        choices=["user", "project"],
        default="user",
        help="Install scope: 'user' (~/.lllm/packages/) or 'project' (lllm_packages/). Default: user.",
    )
    install_parser.add_argument(
        "--no-load",
        action="store_true",
        help="Do not load the package into the current runtime after installing.",
    )

    # ── pkg remove ──────────────────────────────────────────────────────────
    remove_parser = pkg_sub.add_parser("remove", help="Remove an installed package.")
    remove_parser.add_argument("name", help="Package name to remove.")
    remove_parser.add_argument(
        "--scope",
        choices=["user", "project"],
        default=None,
        help="Restrict search to a specific scope. Searches both by default.",
    )

    # ── pkg list ────────────────────────────────────────────────────────────
    list_parser = pkg_sub.add_parser("list", help="List installed packages.")
    list_parser.add_argument(
        "--scope",
        choices=["user", "project"],
        default=None,
        help="Show only packages from this scope.",
    )

    # ── pkg export ──────────────────────────────────────────────────────────
    export_parser = pkg_sub.add_parser(
        "export", help="Export a package to a .zip file."
    )
    export_parser.add_argument("name", help="Package name to export.")
    export_parser.add_argument("output", help="Output path for the .zip file.")
    export_parser.add_argument(
        "--bundle-deps",
        action="store_true",
        help="Bundle all transitive dependencies inside the zip.",
    )

    args = parser.parse_args(argv)

    if args.command == "create":
        try:
            create_project(
                args.project_name, template_ref=args.template, model=args.model
            )
        except Exception as exc:  # pragma: no cover
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "pkg":
        try:
            _handle_pkg(args)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


def _handle_pkg(args: argparse.Namespace) -> None:
    """Dispatch pkg sub-commands."""
    if args.pkg_command == "install":
        _cmd_install(args)
    elif args.pkg_command == "remove":
        _cmd_remove(args)
    elif args.pkg_command == "list":
        _cmd_list(args)
    elif args.pkg_command == "export":
        _cmd_export(args)
    else:
        # No sub-command given for pkg
        print("Usage: lllm pkg {install,remove,list,export} ...", file=sys.stderr)
        sys.exit(1)


def _cmd_install(args: argparse.Namespace) -> None:
    from lllm import install_package

    dest = install_package(
        args.zip_path,
        alias=args.alias,
        scope=args.scope,
        load=not args.no_load,
    )
    alias_note = f" (alias '{args.alias}')" if args.alias else ""
    print(f"Installed{alias_note} to {dest}")


def _cmd_remove(args: argparse.Namespace) -> None:
    from lllm import remove_package

    removed = remove_package(args.name, scope=args.scope)
    print(f"Removed '{args.name}' from {removed}")


def _cmd_list(args: argparse.Namespace) -> None:
    from lllm import list_packages

    packages = list_packages(scope=args.scope)
    if not packages:
        print("No packages installed.")
        return
    # Determine column widths
    col_name = max(len(p["name"]) for p in packages)
    col_ver = max(len(p["version"]) for p in packages) or 7
    col_scope = max(len(p["scope"]) for p in packages)
    header = (
        f"{'NAME':<{col_name}}  {'VERSION':<{col_ver}}  {'SCOPE':<{col_scope}}  PATH"
    )
    print(header)
    print("-" * len(header))
    for p in packages:
        print(
            f"{p['name']:<{col_name}}  {p['version']:<{col_ver}}  {p['scope']:<{col_scope}}  {p['path']}"
        )


def _cmd_export(args: argparse.Namespace) -> None:
    from lllm import export_package, load_runtime

    # Ensure a runtime is loaded so the package is discoverable
    load_runtime()
    out = export_package(args.name, args.output, bundle_deps=args.bundle_deps)
    deps_note = " (with bundled dependencies)" if args.bundle_deps else ""
    print(f"Exported '{args.name}'{deps_note} to {out}")


def create_project(
    project_name: str,
    *,
    template_ref: str = DEFAULT_TEMPLATE,
    model: str = DEFAULT_MODEL,
    cwd: str | Path | None = None,
) -> Path:
    base_dir = Path(cwd) if cwd is not None else Path.cwd()
    target_dir = base_dir / project_name
    source = resolve_template(template_ref)

    project_slug = target_dir.name
    variables = {
        "project_name": project_slug,
        "package_name": normalize_package_name(project_slug),
        "pyproject_name": normalize_pyproject_name(project_slug),
        "project_title": title_from_name(project_slug),
        "model_name": model,
    }

    render_template(source, target_dir, variables)
    print(f"Created project at {target_dir}")
    print(f"Template: {source.spec.name} ({source.kind})")
    print("")
    print("Next steps:")
    print(f"  cd {target_dir}")
    print("  uv sync --extra dev")
    print("  cp .env.example .env")
    print("  uv run python main.py")
    return target_dir


if __name__ == "__main__":
    main()
