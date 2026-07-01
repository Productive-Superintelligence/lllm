# lllm/core/config.py
"""
Package loading and resource discovery.

Entry point: :func:`load_package` — reads ``lllm.toml``, parses
``[package]``, ``[dependencies]``, and all resource sections, then
recursively loads the dependency tree into a package/resource registry.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomllib

from .resource import PackageInfo, ResourceNode
from .runtime import Registry, Runtime, _select_runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORED_FILES = {"__init__.py", "__pycache__"}

PACKAGE_SECTION = "package"
DEPENDENCY_SECTION = "dependencies"
PROMPT_SECTION = "prompts"
PROXY_SECTION = "proxies"
TOOLS_SECTION = "tools"
CONFIG_SECTION = "configs"
TACTIC_SECTION = "tactics"
SERVICE_SECTION = "services"
ASSET_SECTION = "assets"

META_SECTIONS = frozenset({PACKAGE_SECTION, DEPENDENCY_SECTION})
BUILTIN_RESOURCE_SECTIONS = (
    TACTIC_SECTION,
    SERVICE_SECTION,
    CONFIG_SECTION,
    ASSET_SECTION,
    PROMPT_SECTION,
    TOOLS_SECTION,
    PROXY_SECTION,
)
KNOWN_SECTIONS = META_SECTIONS | frozenset(BUILTIN_RESOURCE_SECTIONS)

_SECTION_TO_RESOURCE_TYPE = {
    PROMPT_SECTION: "prompt",
    TOOLS_SECTION: "tool",
    PROXY_SECTION: "proxy",
    CONFIG_SECTION: "config",
    TACTIC_SECTION: "tactic",
    SERVICE_SECTION: "service",
    ASSET_SECTION: "asset",
}

LLLM_CONFIG_ENV = "LLLM_CONFIG"
CONFIG_FILENAMES = ("lllm.toml", ".lllm.toml", "LLLM.toml")
CONFIG_SUBDIRS = ("", "template")


# ---------------------------------------------------------------------------
# TOML entry parsing
# ---------------------------------------------------------------------------


@dataclass
class ParsedPathEntry:
    """Parsed ``[section] paths`` entry.  Supports both string and table forms:

    ``"./dir under vfolder"``  ⟺  ``{path = "./dir", prefix = "vfolder"}``
    ``"./dir under vfolder"``  ⟺  ``{path = "./dir", under = "vfolder"}``
    """

    path: str
    prefix: Optional[str] = None


@dataclass
class ParsedDependencyEntry:
    """Parsed ``[dependencies] packages`` entry.  Supports both forms:

    ``"./pkg as p1"``  ⟺  ``{path = "./pkg", alias = "p1"}``
    ``"./pkg as p1"``  ⟺  ``{path = "./pkg", as = "p1"}``
    """

    path: str
    alias: Optional[str] = None


def _parse_path_entry(entry: Any) -> ParsedPathEntry:
    if isinstance(entry, dict):
        return ParsedPathEntry(
            path=entry["path"],
            prefix=entry.get("prefix") or entry.get("under"),
        )
    if isinstance(entry, str):
        if " under " in entry:
            path_part, pfx = entry.rsplit(" under ", 1)
            return ParsedPathEntry(path=path_part.strip(), prefix=pfx.strip())
        return ParsedPathEntry(path=entry.strip())
    raise ValueError(f"Invalid path entry: {entry!r}")


def _parse_dependency_entry(entry: Any) -> ParsedDependencyEntry:
    if isinstance(entry, dict):
        return ParsedDependencyEntry(
            path=entry["path"],
            alias=entry.get("alias") or entry.get("as"),
        )
    if isinstance(entry, str):
        if " as " in entry:
            path_part, alias = entry.rsplit(" as ", 1)
            return ParsedDependencyEntry(path=path_part.strip(), alias=alias.strip())
        return ParsedDependencyEntry(path=entry.strip())
    raise ValueError(f"Invalid dependency entry: {entry!r}")


# ---------------------------------------------------------------------------
# Config file resolution
# ---------------------------------------------------------------------------


def find_config_file(
    start_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Path]:
    """Locate the nearest ``lllm.toml`` by searching upward."""
    env = os.environ.get(LLLM_CONFIG_ENV)
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            candidate = candidate / "lllm.toml"
        if candidate.is_file():
            return candidate.resolve()

    roots: List[Path] = []
    if start_path is not None:
        roots.append(Path(start_path).resolve())
    roots.append(Path.cwd())

    for root in roots:
        for path in [root, *root.parents]:
            for subdir in CONFIG_SUBDIRS:
                base = path if subdir == "" else path / subdir
                for name in CONFIG_FILENAMES:
                    candidate = base / name
                    if candidate.is_file():
                        return candidate.resolve()
    return None


def load_toml(
    path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load a TOML file.  Stores resolved path in ``data["_config_path"]``."""
    config_path: Optional[Path] = None
    if path:
        p = Path(path).expanduser()
        if p.is_dir():
            p = p / "lllm.toml"
        if p.is_file():
            config_path = p.resolve()
    if config_path is None:
        config_path = find_config_file(path)
    if config_path is None:
        return None
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    data["_config_path"] = config_path
    return data


# ---------------------------------------------------------------------------
# Package loading
# ---------------------------------------------------------------------------


def load_cwd_fallback(
    runtime: Optional[Runtime] = None,
    *,
    registry: Optional[Registry] = None,
) -> bool:
    """Auto-discover standard resource folders in the current working directory.

    Called when no ``lllm.toml`` is found.  Registers the cwd as an anonymous
    package named after the directory, then discovers any of the four built-in
    sections whose default sub-folder exists.

    Returns ``True`` if at least one folder was found and registered.
    """
    runtime = _select_runtime(runtime=runtime, registry=registry)
    cwd = Path.cwd()

    if not any((cwd / s).is_dir() for s in BUILTIN_RESOURCE_SECTIONS):
        return False

    pkg_name = cwd.name
    pkg_info = PackageInfo(
        name=pkg_name,
        version="",
        description="",
        base_dir=str(cwd.resolve()),
    )
    runtime.register_package(pkg_info)
    if runtime._default_namespace is None:
        runtime._default_namespace = pkg_name

    for section_name in BUILTIN_RESOURCE_SECTIONS:
        _discover_section(
            {},
            cwd,
            runtime,
            package_name=pkg_name,
            section_name=section_name,
        )

    return True


def load_package(
    config_path: Optional[str | Path] = None,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
) -> None:
    """Load a package and its dependency tree into a registry.

    Reads ``lllm.toml``, registers the package, loads dependencies
    recursively, then discovers resources in every section.
    """
    runtime = _select_runtime(runtime=runtime, registry=registry)
    config = load_toml(config_path)
    if not config:
        return

    base_dir = Path(config["_config_path"]).parent
    abs_base = str(base_dir.resolve())

    # Cycle detection
    if abs_base in runtime._loaded_package_paths:
        logger.debug("Package at %s already loaded, skipping", abs_base)
        return
    runtime._loaded_package_paths.add(abs_base)

    # [package]
    pkg_section = config.get(PACKAGE_SECTION, {})
    pkg_name = pkg_section.get("name", base_dir.name)
    pkg_info = PackageInfo(
        name=pkg_name,
        version=pkg_section.get("version", ""),
        description=pkg_section.get("description", ""),
        base_dir=abs_base,
    )
    runtime.register_package(pkg_info)

    if runtime._default_namespace is None:
        runtime._default_namespace = pkg_name

    # [dependencies]
    _load_dependencies(config, base_dir, runtime)

    # Built-in resource sections
    for section_name in BUILTIN_RESOURCE_SECTIONS:
        _discover_section(
            config.get(section_name, {}),
            base_dir,
            runtime,
            package_name=pkg_name,
            section_name=section_name,
        )

    # Custom sections
    for section_name, section_data in config.items():
        if section_name.startswith("_") or section_name in KNOWN_SECTIONS:
            continue
        if not isinstance(section_data, dict):
            continue
        _discover_section(
            section_data,
            base_dir,
            runtime,
            package_name=pkg_name,
            section_name=section_name,
        )


def _load_dependencies(config: Dict, base_dir: Path, runtime: Runtime) -> None:
    deps = config.get(DEPENDENCY_SECTION, {}).get("packages", [])
    for raw in deps:
        parsed = _parse_dependency_entry(raw)
        dep_dir = (base_dir / parsed.path).resolve()

        dep_toml = None
        for name in CONFIG_FILENAMES:
            candidate = dep_dir / name
            if candidate.exists():
                dep_toml = candidate
                break
        if dep_toml is None:
            warnings.warn(
                f"Dependency '{parsed.path}' has no lllm.toml at {dep_dir}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        load_package(str(dep_toml), runtime=runtime)

        if parsed.alias:
            dep_config = load_toml(str(dep_toml))
            if dep_config:
                dep_name = dep_config.get(PACKAGE_SECTION, {}).get("name", dep_dir.name)
                if dep_name in runtime.packages:
                    original = runtime.packages[dep_name]
                    aliased = PackageInfo(
                        name=original.name,
                        version=original.version,
                        description=original.description,
                        base_dir=original.base_dir,
                        alias=parsed.alias,
                    )
                    runtime.packages[parsed.alias] = aliased
                    _alias_package_resources(runtime, dep_name, parsed.alias)


def _alias_package_resources(runtime: Runtime, original_name: str, alias: str) -> None:
    """Re-register all resources from *original_name* under *alias*."""
    prefix = f"{original_name}."
    to_alias: List[Tuple[str, ResourceNode]] = []

    for qk, node in list(runtime._resources.items()):
        if not node.namespace.startswith(prefix):
            continue
        section_part = node.namespace[len(prefix) :]
        new_ns = f"{alias}.{section_part}"
        to_alias.append((new_ns, node))

    for new_ns, node in to_alias:
        if node.is_loaded:
            alias_node = ResourceNode.eager(
                node.key,
                node.value,
                namespace=new_ns,
                resource_type=node.resource_type,
            )
        else:
            alias_node = ResourceNode.lazy(
                node.key,
                node._loader,
                namespace=new_ns,
                resource_type=node.resource_type,
            )
        runtime.register(alias_node, overwrite=True)


# ---------------------------------------------------------------------------
# Section discovery
# ---------------------------------------------------------------------------


def _discover_section(
    section: dict,
    base_dir: Path,
    runtime: Runtime,
    package_name: str,
    section_name: str,
) -> None:
    raw_entries = section.get("paths") or []

    # Default subfolder fallback
    if not raw_entries:
        default = base_dir / section_name
        if default.is_dir():
            raw_entries = [str(default)]
        else:
            return

    resource_type = _SECTION_TO_RESOURCE_TYPE.get(section_name, section_name)
    namespace = f"{package_name}.{section_name}"

    for raw in raw_entries:
        parsed = _parse_path_entry(raw)
        path = Path(parsed.path)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            warnings.warn(
                f"LLLM discovery skipped missing path: {path}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue

        prefix = parsed.prefix or ""

        if section_name == CONFIG_SECTION:
            _discover_configs(path, runtime, namespace, resource_type, prefix)
        elif section_name == TOOLS_SECTION:
            # Keep [tools] useful for colocated metadata files while also
            # registering Python @tool Function objects as callable resources.
            _discover_files(path, runtime, namespace, resource_type, prefix)
            _discover_python_modules(
                path, runtime, namespace, section_name, resource_type, prefix
            )
        elif section_name == SERVICE_SECTION:
            _discover_services(path, runtime, namespace, resource_type, prefix)
            _discover_python_modules(
                path, runtime, namespace, section_name, resource_type, prefix
            )
        elif section_name == ASSET_SECTION:
            _discover_files(path, runtime, namespace, resource_type, prefix)
        elif section_name in {PROMPT_SECTION, PROXY_SECTION, TACTIC_SECTION}:
            # Built-in Python-based sections (prompts, proxies, tactics)
            _discover_python_modules(
                path, runtime, namespace, section_name, resource_type, prefix
            )
        else:
            # Custom section: discover all files (lazy), PLUS any .py modules
            _discover_files(path, runtime, namespace, resource_type, prefix)
            _discover_python_modules(
                path, runtime, namespace, section_name, resource_type, prefix
            )


# ---------------------------------------------------------------------------
# Generic file discovery (custom sections — images, models, JSON, etc.)
# ---------------------------------------------------------------------------

# Files that are already handled by _discover_python_modules or should be skipped
_SKIP_EXTENSIONS = {".py", ".pyc", ".pyo"}

# Known structured formats that get a typed loader instead of raw bytes
_STRUCTURED_LOADERS = {
    ".json": "_json",
    ".yaml": "_yaml",
    ".yml": "_yaml",
    ".toml": "_toml",
}


def _load_json(p: Path):
    import json

    with p.open() as f:
        return json.load(f)


def _load_yaml(p: Path):
    import yaml

    with p.open() as f:
        return yaml.safe_load(f)


def _load_toml(p: Path):
    import tomllib

    with p.open("rb") as f:
        return tomllib.load(f)


_LOADER_FUNCS = {
    "_json": _load_json,
    "_yaml": _load_yaml,
    "_toml": _load_toml,
}


def _discover_files(
    root: Path,
    runtime: Runtime,
    namespace: str,
    resource_type: str,
    prefix: str,
) -> None:
    """Discover arbitrary files and register them as lazy ``ResourceNode``s.

    Used for custom sections (``[assets]``, ``[models]``, etc.).

    Loader behavior by extension:
        - ``.json`` → parsed as dict/list via ``json.load``
        - ``.yaml`` / ``.yml`` → parsed via ``yaml.safe_load``
        - ``.toml`` → parsed via ``tomllib.load``
        - Everything else → loaded as raw ``bytes``

    The key **includes the file extension** (unlike Python-based discovery
    where ``.py`` is stripped), because the extension is part of the file
    identity — ``logo.png`` and ``logo.svg`` are different resources.

    Each node also stores the absolute file path in
    ``metadata["file_path"]`` so users can load the file differently if
    the default loader doesn't suit their needs::

        node = runtime.get_node("my_pkg.assets:models/classifier.pt")
        path = node.metadata["file_path"]   # use your own loader
    """
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix in _SKIP_EXTENSIONS:
            continue
        if f.name.startswith("_") or f.name.startswith("."):
            continue
        if "__pycache__" in f.parts:
            continue

        relative = str(f.relative_to(root)).replace(os.sep, "/")
        key = f"{prefix}/{relative}".strip("/")
        file_path = f  # capture for closure
        abs_path = str(f.resolve())

        # Pick the right loader
        ext = f.suffix.lower()
        if ext in _STRUCTURED_LOADERS:
            loader_key = _STRUCTURED_LOADERS[ext]
            loader_func = _LOADER_FUNCS[loader_key]

            def _loader(p=file_path, load=loader_func):
                return load(p)
        else:

            def _loader(p=file_path):
                return p.read_bytes()

        node = ResourceNode.lazy(
            key,
            _loader,
            namespace=namespace,
            resource_type=resource_type,
            file_path=abs_path,
        )
        try:
            runtime.register(node, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to register file '%s': %s", key, exc)


# ---------------------------------------------------------------------------
# Python module discovery
# ---------------------------------------------------------------------------


def _discover_python_modules(
    root: Path,
    runtime: Runtime,
    namespace: str,
    section_name: str,
    resource_type: str,
    prefix: str,
) -> None:
    for py_file in sorted(root.rglob("*.py")):
        if py_file.name in IGNORED_FILES or py_file.name.startswith("_"):
            continue
        if "__pycache__" in py_file.parts:
            continue

        relative = str(py_file.relative_to(root).with_suffix("")).replace(os.sep, "/")
        mod_ns = f"lllm._discovered.{namespace}.{relative.replace('/', '.')}"

        try:
            module = _load_module(py_file, mod_ns)
        except Exception as exc:
            warnings.warn(
                f"LLLM discovery failed to load {py_file}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if section_name == PROMPT_SECTION:
            _register_prompts(
                module, relative, runtime, namespace, resource_type, prefix
            )
        elif section_name == TOOLS_SECTION:
            _register_tools(module, relative, runtime, namespace, resource_type, prefix)
            # Preserve the previous custom-section behavior for packages that
            # already used [tools] to group non-Function Python resources.
            _register_prompts(
                module, relative, runtime, namespace, resource_type, prefix
            )
            _register_proxies(
                module, relative, runtime, namespace, resource_type, prefix
            )
            _register_tactics(
                module, relative, runtime, namespace, resource_type, prefix
            )
        elif section_name == PROXY_SECTION:
            _register_proxies(
                module, relative, runtime, namespace, resource_type, prefix
            )
        elif section_name == TACTIC_SECTION:
            _register_tactics(
                module, relative, runtime, namespace, resource_type, prefix
            )
        elif section_name == SERVICE_SECTION:
            _register_services(
                module, relative, runtime, namespace, resource_type, prefix
            )
        else:
            # Custom section — try all typed registrations
            _register_tools(module, relative, runtime, namespace, resource_type, prefix)
            _register_prompts(
                module, relative, runtime, namespace, resource_type, prefix
            )
            _register_proxies(
                module, relative, runtime, namespace, resource_type, prefix
            )
            _register_tactics(
                module, relative, runtime, namespace, resource_type, prefix
            )
            _register_services(
                module, relative, runtime, namespace, resource_type, prefix
            )


def _discover_configs(
    root: Path,
    runtime: Runtime,
    namespace: str,
    resource_type: str,
    prefix: str,
) -> None:
    for pattern in ("**/*.yaml", "**/*.yml"):
        for f in sorted(
            root.rglob(pattern.split("/")[-1])
            if "/" not in pattern
            else root.glob(pattern)
        ):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root).with_suffix("")).replace(os.sep, "/")
            key = f"{prefix}/{rel}".strip("/")
            file_path = f  # capture for closure

            def _loader(p=file_path):
                import yaml

                with p.open() as fh:
                    return yaml.safe_load(fh)

            node = ResourceNode.lazy(
                key, _loader, namespace=namespace, resource_type=resource_type
            )
            try:
                runtime.register(node, overwrite=True)
            except Exception as exc:
                logger.warning("Failed to register config '%s': %s", key, exc)


def _discover_services(
    root: Path,
    runtime: Runtime,
    namespace: str,
    resource_type: str,
    prefix: str,
) -> None:
    """Discover service manifests as lazy platform resources."""

    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".json", ".toml", ".yaml", ".yml"}:
            continue
        if f.name.startswith("_") or f.name.startswith("."):
            continue
        if "__pycache__" in f.parts:
            continue

        rel = str(f.relative_to(root).with_suffix("")).replace(os.sep, "/")
        key = f"{prefix}/{rel}".strip("/")
        file_path = f
        abs_path = str(f.resolve())

        def _loader(p=file_path):
            from ..server.spec import load_service

            return load_service(p)

        node = ResourceNode.lazy(
            key,
            _loader,
            namespace=namespace,
            resource_type=resource_type,
            file_path=abs_path,
        )
        try:
            runtime.register(node, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to register service '%s': %s", key, exc)


def _load_module(file_path: Path, namespace: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(namespace, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Typed registration helpers
# ---------------------------------------------------------------------------


def _make_key(prefix: str, relative: str, name: str) -> str:
    return "/".join(p for p in [prefix, relative, name] if p).strip("/")


def _register_prompts(module, relative, runtime, namespace, resource_type, prefix):
    from .prompt import Prompt

    for attr_name, attr in vars(module).items():
        if not isinstance(attr, Prompt):
            continue
        key = _make_key(prefix, relative, attr.path)
        node = ResourceNode.eager(
            key, attr, namespace=namespace, resource_type=resource_type
        )
        try:
            runtime.register(node, overwrite=True)
            attr._qualified_key = node.qualified_key
            attr._resource_namespace = node.namespace
        except Exception as exc:
            logger.warning("Failed to register prompt '%s': %s", key, exc)


def _register_tools(module, relative, runtime, namespace, resource_type, prefix):
    from .prompt import Function

    for attr_name, attr in vars(module).items():
        if not isinstance(attr, Function):
            continue
        if attr.name == Path(relative).name:
            key = _make_key(prefix, "", relative)
        else:
            key = _make_key(prefix, relative, attr.name)
        node = ResourceNode.eager(
            key, attr, namespace=namespace, resource_type=resource_type
        )
        try:
            runtime.register(node, overwrite=True)
            attr._qualified_key = node.qualified_key
            attr._resource_namespace = node.namespace
        except Exception as exc:
            logger.warning("Failed to register tool '%s': %s", key, exc)


def _register_proxies(module, relative, runtime, namespace, resource_type, prefix):
    from ..proxies.base import BaseProxy

    for attr_name, cls in vars(module).items():
        if not (
            inspect.isclass(cls) and issubclass(cls, BaseProxy) and cls is not BaseProxy
        ):
            continue
        proxy_path = getattr(cls, "_proxy_path", None)
        if proxy_path:
            key = f"{prefix}/{proxy_path}".strip("/") if prefix else proxy_path
        else:
            key = _make_key(prefix, relative, cls.__name__)
        node = ResourceNode.eager(
            key, cls, namespace=namespace, resource_type=resource_type
        )
        try:
            runtime.register(node, overwrite=True)
            cls._qualified_key = node.qualified_key
            cls._resource_namespace = node.namespace
        except Exception as exc:
            logger.warning("Failed to register proxy '%s': %s", key, exc)


def _register_tactics(module, relative, runtime, namespace, resource_type, prefix):
    from .tactic import Tactic

    for attr_name, cls in vars(module).items():
        if not (
            inspect.isclass(cls)
            and issubclass(cls, Tactic)
            and cls is not Tactic
        ):
            continue
        tactic_name = getattr(cls, "name", None)
        if not tactic_name:
            continue
        key = f"{prefix}/{tactic_name}".strip("/") if prefix else tactic_name
        node = ResourceNode.eager(
            key, cls, namespace=namespace, resource_type=resource_type
        )
        try:
            runtime.register(node, overwrite=True)
            cls._qualified_key = node.qualified_key
            cls._resource_namespace = node.namespace
        except Exception as exc:
            logger.warning("Failed to register tactic '%s': %s", key, exc)


def _register_services(module, relative, runtime, namespace, resource_type, prefix):
    from ..server.spec import Service

    for attr_name, attr in vars(module).items():
        if not isinstance(attr, Service):
            continue
        key = _make_key(prefix, relative, attr.name)
        node = ResourceNode.eager(
            key, attr, namespace=namespace, resource_type=resource_type
        )
        try:
            runtime.register(node, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to register service '%s': %s", key, exc)


# ---------------------------------------------------------------------------
# Config resolution (inheritance via `base` key)
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base*.

    - Dict values are merged recursively.
    - List values are replaced (not appended).
    - Scalar values are replaced.

    Neither input is mutated; returns a new dict.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def resolve_config(
    name: str,
    runtime: Optional[Runtime] = None,
    *,
    registry: Optional[Registry] = None,
    _visited: Optional[set] = None,
) -> Dict[str, Any]:
    """Load a config by name and resolve ``base`` inheritance.

    The ``base`` key points to another config name (no ``.yaml`` suffix).
    Inheritance is recursive — each level's keys override the parent's,
    with dict values merged deeply (so ``model_args`` from both parent
    and child are combined, not replaced wholesale).

    Parameters
    ----------
    name:
        Config resource name, e.g. ``"default"`` or
        ``"agent_cfgs/agent1"`` or ``"my_pkg:default"``.
    runtime:
        Compatibility name for the registry to look up configs from.
    registry:
        Registry to look up configs from.

    Returns
    -------
    The fully merged config dict (``base`` key removed).
    """
    runtime = _select_runtime(runtime=runtime, registry=registry)
    _visited = _visited or set()

    if name in _visited:
        raise ValueError(
            f"Circular config inheritance detected: "
            f"'{name}' already in chain {_visited}"
        )
    _visited.add(name)

    config = runtime.get_config(name)
    if not isinstance(config, dict):
        raise TypeError(f"Config '{name}' is not a dict (got {type(config).__name__})")

    config = config.copy()
    base_name = config.pop("base", None)
    if base_name is None:
        return config

    parent = resolve_config(base_name, registry=runtime, _visited=_visited)
    return _deep_merge(parent, config)


def vendor_config(
    source: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
) -> Dict[str, Any]:
    """Resolve a dependency's config and optionally apply overrides.

    Use this to "vendor" a dependency's config into your own package —
    materializing it into a self-contained dict with your overrides
    applied on top.

    Parameters
    ----------
    source:
        Config name to resolve, e.g. ``"A:default"`` or ``"default"``.
    overrides:
        Optional dict of overrides to deep-merge on top of the resolved
        config.  Dict values merge recursively (so you can override a
        single ``model_args`` key without losing the rest).
    runtime:
        Compatibility name for the registry to look up configs from.
    registry:
        Registry to look up configs from.

    Returns
    -------
    A fully materialized dict (no ``base`` key) with overrides applied.

    Example
    -------
    ::

        # Pull package A's config and pin model choice
        cfg = vendor_config("A:default", {
            "global": {
                "model_name": "gpt-4o",
                "model_args": {"temperature": 0.05},
            },
        })

        # Register as your own config
        runtime.register_config("vendor/A", cfg, namespace="my_pkg.configs")

        # Or save to disk
        import yaml
        with open("configs/vendor/A.yaml", "w") as f:
            yaml.dump(cfg, f)
    """
    config = resolve_config(source, runtime=runtime, registry=registry)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


_NATIVE_CONFIG_EXPORTS = frozenset(
    {
        "AgentSpec",
        "ContextManagerConfig",
        "ProxyConfig",
        "SkillsConfig",
        "parse_agent_configs",
    }
)


def __getattr__(name: str) -> Any:
    """Lazy compatibility access for native runtime config classes."""

    if name in _NATIVE_CONFIG_EXPORTS:
        from . import native_config

        return getattr(native_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
