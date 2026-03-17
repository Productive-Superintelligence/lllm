from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
}

PLACEHOLDERS = {
    "__project_name__": "{name}",
    "{{project_name}}": "{name}",
    "{{PROJECT_NAME}}": "{name_upper}",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="lllm", description="LLLM helper CLI.")
    subparsers = parser.add_subparsers(dest="command")

    # ── create ──────────────────────────────────────────────────────────────
    create_parser = subparsers.add_parser("create", help="Create a new LLLM project scaffold.")
    create_parser.add_argument(
        "--name",
        default="system",
        help="Name of the project folder to create (default: system).",
    )
    create_parser.add_argument(
        "--template",
        default="init_template",
        help="Template folder inside templates/ to use (default: init_template).",
    )

    # ── pkg install ──────────────────────────────────────────────────────────
    pkg_parser = subparsers.add_parser("pkg", help="Package management commands.")
    pkg_sub = pkg_parser.add_subparsers(dest="pkg_command")

    install_parser = pkg_sub.add_parser("install", help="Install a package from a .zip file.")
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
    export_parser = pkg_sub.add_parser("export", help="Export a package to a .zip file.")
    export_parser.add_argument("name", help="Package name to export.")
    export_parser.add_argument("output", help="Output path for the .zip file.")
    export_parser.add_argument(
        "--bundle-deps",
        action="store_true",
        help="Bundle all transitive dependencies inside the zip.",
    )

    args = parser.parse_args()

    if args.command == "create":
        try:
            create_project(args.name, args.template)
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
        import argparse as _ap
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
    header = f"{'NAME':<{col_name}}  {'VERSION':<{col_ver}}  {'SCOPE':<{col_scope}}  PATH"
    print(header)
    print("-" * len(header))
    for p in packages:
        print(f"{p['name']:<{col_name}}  {p['version']:<{col_ver}}  {p['scope']:<{col_scope}}  {p['path']}")


def _cmd_export(args: argparse.Namespace) -> None:
    from lllm import export_package, load_runtime
    # Ensure a runtime is loaded so the package is discoverable
    load_runtime()
    out = export_package(args.name, args.output, bundle_deps=args.bundle_deps)
    deps_note = " (with bundled dependencies)" if args.bundle_deps else ""
    print(f"Exported '{args.name}'{deps_note} to {out}")


def create_project(name: str, template_name: str) -> None:
    target_dir = Path.cwd() / name
    if target_dir.exists():
        raise FileExistsError(f"Path '{target_dir}' already exists.")

    template_dir = _resolve_template(template_name)
    if template_dir is None:
        raise FileNotFoundError(f"Template '{template_name}' not found.")

    replacements = {
        key: value.format(name=name, name_upper=name.upper())
        for key, value in PLACEHOLDERS.items()
    }
    _copy_template(template_dir, target_dir, replacements)
    print(f"Created project at {target_dir}")


def _resolve_template(template_name: str) -> Path | None:
    package_root = Path(__file__).resolve().parent.parent
    repo_template = package_root / "templates" / template_name
    if repo_template.exists():
        return repo_template
    return None


def _copy_template(src_dir: Path, dst_dir: Path, replacements: Dict[str, str]) -> None:
    for path in src_dir.rglob("*"):
        relative = path.relative_to(src_dir)
        rendered_relative = _render_path(relative, replacements)
        target_path = dst_dir / rendered_relative
        if path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in TEXT_EXTENSIONS:
            content = path.read_text(encoding="utf-8")
            for placeholder, value in replacements.items():
                content = content.replace(placeholder, value)
            target_path.write_text(content, encoding="utf-8")
        else:
            shutil.copy2(path, target_path)


def _render_path(path: Path, replacements: Dict[str, str]) -> Path:
    parts = []
    for part in path.parts:
        new_part = part
        for placeholder, value in replacements.items():
            new_part = new_part.replace(placeholder, value)
        parts.append(new_part)
    return Path(*parts)


if __name__ == "__main__":
    main()
