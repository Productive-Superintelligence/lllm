# lllm/core/config.py
"""
Package loading and resource discovery.

Entry point: :func:`load_package` — reads ``lllm.toml``, parses
``[package]``, ``[dependencies]``, and all resource sections, then
recursively loads the dependency tree into a :class:`Runtime`.
"""
from __future__ import annotations

import datetime as dt
import os
import importlib.util
import inspect
import logging
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, List, Dict, Tuple
import tomllib

from lllm.core.runtime import Runtime, get_default_runtime
from lllm.core.resource import ResourceNode, PackageInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORED_FILES = {"__init__.py", "__pycache__"}

PACKAGE_SECTION = "package"
DEPENDENCY_SECTION = "dependencies"
PROMPT_SECTION = "prompts"
PROXY_SECTION = "proxies"
CONFIG_SECTION = "configs"
TACTIC_SECTION = "tactics"

META_SECTIONS = frozenset({PACKAGE_SECTION, DEPENDENCY_SECTION})
BUILTIN_RESOURCE_SECTIONS = (PROMPT_SECTION, PROXY_SECTION, CONFIG_SECTION, TACTIC_SECTION)
KNOWN_SECTIONS = META_SECTIONS | frozenset(BUILTIN_RESOURCE_SECTIONS)

_SECTION_TO_RESOURCE_TYPE = {
    PROMPT_SECTION: "prompt",
    PROXY_SECTION: "proxy",
    CONFIG_SECTION: "config",
    TACTIC_SECTION: "tactic",
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


def load_toml(path: Optional[str | os.PathLike[str]] = None) -> Optional[Dict[str, Any]]:
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

def load_cwd_fallback(runtime: Optional[Runtime] = None) -> bool:
    """Auto-discover standard resource folders in the current working directory.

    Called when no ``lllm.toml`` is found.  Registers the cwd as an anonymous
    package named after the directory, then discovers any of the four built-in
    sections whose default sub-folder exists.

    Returns ``True`` if at least one folder was found and registered.
    """
    runtime = runtime or get_default_runtime()
    cwd = Path.cwd()

    if not any((cwd / s).is_dir() for s in BUILTIN_RESOURCE_SECTIONS):
        return False

    pkg_name = cwd.name
    pkg_info = PackageInfo(
        name=pkg_name, version="", description="",
        base_dir=str(cwd.resolve()),
    )
    runtime.register_package(pkg_info)
    if runtime._default_namespace is None:
        runtime._default_namespace = pkg_name

    for section_name in BUILTIN_RESOURCE_SECTIONS:
        _discover_section(
            {}, cwd, runtime,
            package_name=pkg_name, section_name=section_name,
        )

    return True


def load_package(
    config_path: Optional[str | Path] = None,
    *,
    runtime: Optional[Runtime] = None,
) -> None:
    """Load a package and its dependency tree into a runtime.

    Reads ``lllm.toml``, registers the package, loads dependencies
    recursively, then discovers resources in every section.
    """
    runtime = runtime or get_default_runtime()
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
            config.get(section_name, {}), base_dir, runtime,
            package_name=pkg_name, section_name=section_name,
        )

    # Custom sections
    for section_name, section_data in config.items():
        if section_name.startswith("_") or section_name in KNOWN_SECTIONS:
            continue
        if not isinstance(section_data, dict):
            continue
        _discover_section(
            section_data, base_dir, runtime,
            package_name=pkg_name, section_name=section_name,
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
            warnings.warn(f"Dependency '{parsed.path}' has no lllm.toml at {dep_dir}",
                          RuntimeWarning, stacklevel=2)
            continue

        load_package(str(dep_toml), runtime=runtime)

        if parsed.alias:
            dep_config = load_toml(str(dep_toml))
            if dep_config:
                dep_name = dep_config.get(PACKAGE_SECTION, {}).get("name", dep_dir.name)
                if dep_name in runtime.packages:
                    original = runtime.packages[dep_name]
                    aliased = PackageInfo(
                        name=original.name, version=original.version,
                        description=original.description, base_dir=original.base_dir,
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
        section_part = node.namespace[len(prefix):]
        new_ns = f"{alias}.{section_part}"
        to_alias.append((new_ns, node))

    for new_ns, node in to_alias:
        if node.is_loaded:
            alias_node = ResourceNode.eager(
                node.key, node.value, namespace=new_ns,
                resource_type=node.resource_type,
            )
        else:
            alias_node = ResourceNode.lazy(
                node.key, node._loader, namespace=new_ns,
                resource_type=node.resource_type,
            )
        runtime.register(alias_node, overwrite=True)


# ---------------------------------------------------------------------------
# Section discovery
# ---------------------------------------------------------------------------

def _discover_section(
    section: dict, base_dir: Path, runtime: Runtime,
    package_name: str, section_name: str,
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
            warnings.warn(f"LLLM discovery skipped missing path: {path}",
                          RuntimeWarning, stacklevel=3)
            continue

        prefix = parsed.prefix or ""

        if section_name == CONFIG_SECTION:
            _discover_configs(path, runtime, namespace, resource_type, prefix)
        elif section_name in BUILTIN_RESOURCE_SECTIONS:
            # Built-in Python-based sections (prompts, proxies, tactics)
            _discover_python_modules(path, runtime, namespace, section_name,
                                     resource_type, prefix)
        else:
            # Custom section: discover all files (lazy), PLUS any .py modules
            _discover_files(path, runtime, namespace, resource_type, prefix)
            _discover_python_modules(path, runtime, namespace, section_name,
                                     resource_type, prefix)


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
            key, _loader,
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
    root: Path, runtime: Runtime, namespace: str,
    section_name: str, resource_type: str, prefix: str,
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
            warnings.warn(f"LLLM discovery failed to load {py_file}: {exc}",
                          RuntimeWarning, stacklevel=2)
            continue

        if section_name == PROMPT_SECTION:
            _register_prompts(module, relative, runtime, namespace, resource_type, prefix)
        elif section_name == PROXY_SECTION:
            _register_proxies(module, relative, runtime, namespace, resource_type, prefix)
        elif section_name == TACTIC_SECTION:
            _register_tactics(module, relative, runtime, namespace, resource_type, prefix)
        else:
            # Custom section — try all typed registrations
            _register_prompts(module, relative, runtime, namespace, resource_type, prefix)
            _register_proxies(module, relative, runtime, namespace, resource_type, prefix)
            _register_tactics(module, relative, runtime, namespace, resource_type, prefix)


def _discover_configs(
    root: Path, runtime: Runtime, namespace: str,
    resource_type: str, prefix: str,
) -> None:
    for pattern in ("**/*.yaml", "**/*.yml"):
        for f in sorted(root.rglob(pattern.split("/")[-1]) if "/" not in pattern
                        else root.glob(pattern)):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root).with_suffix("")).replace(os.sep, "/")
            key = f"{prefix}/{rel}".strip("/")
            file_path = f  # capture for closure

            def _loader(p=file_path):
                import yaml
                with p.open() as fh:
                    return yaml.safe_load(fh)

            node = ResourceNode.lazy(key, _loader, namespace=namespace,
                                     resource_type=resource_type)
            try:
                runtime.register(node, overwrite=True)
            except Exception as exc:
                logger.warning("Failed to register config '%s': %s", key, exc)


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
    from lllm.core.prompt import Prompt
    for attr_name, attr in vars(module).items():
        if not isinstance(attr, Prompt):
            continue
        key = _make_key(prefix, relative, attr.path)
        node = ResourceNode.eager(key, attr, namespace=namespace,
                                  resource_type=resource_type)
        try:
            runtime.register(node, overwrite=True)
            attr._qualified_key = node.qualified_key
        except Exception as exc:
            logger.warning("Failed to register prompt '%s': %s", key, exc)


def _register_proxies(module, relative, runtime, namespace, resource_type, prefix):
    from lllm.proxies.base import BaseProxy
    for attr_name, cls in vars(module).items():
        if not (inspect.isclass(cls) and issubclass(cls, BaseProxy) and cls is not BaseProxy):
            continue
        proxy_path = getattr(cls, "_proxy_path", None)
        if proxy_path:
            key = f"{prefix}/{proxy_path}".strip("/") if prefix else proxy_path
        else:
            key = _make_key(prefix, relative, cls.__name__)
        node = ResourceNode.eager(key, cls, namespace=namespace,
                                  resource_type=resource_type)
        try:
            runtime.register(node, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to register proxy '%s': %s", key, exc)


def _register_tactics(module, relative, runtime, namespace, resource_type, prefix):
    from abc import ABC
    for attr_name, cls in vars(module).items():
        if not (inspect.isclass(cls) and issubclass(cls, ABC)):
            continue
        tactic_name = getattr(cls, "name", None)
        if not tactic_name or not hasattr(cls, "agent_group"):
            continue
        key = f"{prefix}/{tactic_name}".strip("/") if prefix else tactic_name
        node = ResourceNode.eager(key, cls, namespace=namespace,
                                  resource_type=resource_type)
        try:
            runtime.register(node, overwrite=True)
        except Exception as exc:
            logger.warning("Failed to register tactic '%s': %s", key, exc)


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
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def resolve_config(
    name: str,
    runtime: Optional[Runtime] = None,
    *,
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
        The runtime to look up configs from.

    Returns
    -------
    The fully merged config dict (``base`` key removed).
    """
    runtime = runtime or get_default_runtime()
    _visited = _visited or set()

    if name in _visited:
        raise ValueError(
            f"Circular config inheritance detected: "
            f"'{name}' already in chain {_visited}"
        )
    _visited.add(name)

    config = runtime.get_config(name)
    if not isinstance(config, dict):
        raise TypeError(
            f"Config '{name}' is not a dict (got {type(config).__name__})"
        )

    config = config.copy()
    base_name = config.pop("base", None)
    if base_name is None:
        return config

    parent = resolve_config(base_name, runtime, _visited=_visited)
    return _deep_merge(parent, config)


def vendor_config(
    source: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    runtime: Optional[Runtime] = None,
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
        The runtime to look up configs from.

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
    config = resolve_config(source, runtime)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


# ---------------------------------------------------------------------------
# AgentSpec — config → agent intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class ProxyConfig:
    """
    Configuration for proxy-based tool calling on an agent.

    Settable globally (under the ``global`` key in tactic config) and
    overridable per-agent.  When both are present, the per-agent dict is
    deep-merged on top of the global one, so agents can override individual
    fields (e.g. swap ``exec_env``) without repeating everything.

    Config format (YAML)::

        proxy:
          activate_proxies: [fmp, fred]   # which proxies to load; empty = all
          deploy_mode: false               # passed through to proxy instances
          cutoff_date: "2024-01-01"        # ISO date; restricts data range
          exec_env: interpreter      # "interpreter" | "jupyter" | null
          max_output_chars: 5000           # truncate run_python output (interpreter only)
          truncation_indicator: "... (truncated)"
          timeout: 60.0                    # seconds before TimeoutError (interpreter only)
          prompt_template: null            # override auto-selected system-prompt block

    **exec_env values**

    ``"interpreter"`` (default)
        Agent calls ``run_python`` tool.  LLLM runs code in a lightweight
        in-process :class:`~lllm.proxies.interpreter.AgentInterpreter` with a
        persistent namespace.  Parallel-safe, zero subprocess overhead.

    ``"jupyter"``
        Agent writes ``<python_cell>`` / ``<markdown_cell>`` XML tags.  Your
        tactic extracts these and runs them via
        :class:`~lllm.sandbox.jupyter.JupyterSession`.  Only ``query_api_doc``
        is injected as a tool; ``run_python`` is **not** added.

    ``null`` (or any unrecognised string)
        No execution tool injected.  Useful when the agent only needs API
        awareness (``query_api_doc`` + directory in prompt) but execution is
        handled externally or not needed at all.

    Future sandbox types (e.g. ``"docker"``, ``"wasm"``) can be added by
    extending the tactic — the string passes through without validation.
    """

    activate_proxies: List[str] = field(default_factory=list)
    deploy_mode: bool = False
    cutoff_date: Optional[str] = None           # ISO date string e.g. "2024-01-01"
    exec_env: Optional[str] = "interpreter"  # "interpreter" | "jupyter" | None
    max_output_chars: int = 5000
    truncation_indicator: str = "... (truncated)"
    timeout: float = 60.0
    prompt_template: Optional[str] = None       # None → auto-select based on exec_env

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProxyConfig":
        return cls(
            activate_proxies=d.get("activate_proxies", []),
            deploy_mode=d.get("deploy_mode", False),
            cutoff_date=d.get("cutoff_date", None),
            exec_env=d.get("exec_env", "interpreter"),
            max_output_chars=d.get("max_output_chars", 5000),
            truncation_indicator=d.get("truncation_indicator", "... (truncated)"),
            timeout=d.get("timeout", 60.0),
            prompt_template=d.get("prompt_template", None),
        )


@dataclass
class ContextManagerConfig:
    """
    Configuration for context-window management on an agent.

    Settable globally (under the ``global`` key in tactic config) and
    overridable per-agent.  When both are present the per-agent dict is
    deep-merged on top of the global one.

    Config format (YAML)::

        context_manager:
          type: default       # "default" → DefaultContextManager; null → disabled
          max_tokens: 128000  # optional hard cap; omit to auto-detect from litellm

    **type values**

    ``"default"`` (built-in)
        Uses :class:`~lllm.core.dialog.DefaultContextManager`: drops/truncates
        old messages so total tokens stay within the model's context window.

    Custom string (e.g. ``"summary"``)
        Looked up in the runtime via
        :meth:`~lllm.core.runtime.Runtime.get_context_manager`.  Register your
        class first::

            runtime.register_context_manager("summary", SummaryCompressor)

    ``null`` / omitted
        Context management disabled for this agent.
    """

    type: str = "default"
    max_tokens: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextManagerConfig":
        return cls(
            type=d.get("type", "default"),
            max_tokens=d.get("max_tokens", None),
        )

    def build(self, model_name: str, runtime: Runtime):
        """Instantiate and return the configured :class:`~lllm.core.dialog.ContextManager`."""
        from lllm.core.dialog import DefaultContextManager
        if self.type in (None, "null", "none"):
            return None
        if self.type == "default":
            return DefaultContextManager(model_name=model_name, max_tokens=self.max_tokens)
        # Custom type registered in the runtime
        cm_cls = runtime.get_context_manager(self.type)
        return cm_cls(model_name=model_name, max_tokens=self.max_tokens)


_KNOWN_AGENT_KEYS = frozenset({
    "name", "model_name", "system_prompt", "system_prompt_path",
    "api_type", "model_args",
    "max_exception_retry", "max_interrupt_steps", "max_llm_recall",
    "extra_settings", "proxy", "context_manager",
})


@dataclass
class AgentSpec:
    """
    Parsed, validated description of one agent from config.

    Intermediate representation between raw YAML and live Agent instances.
    Config parsing fails here with clear errors; Agent construction is trivial.

    Config format (per-agent, after global merge)::

        name: analyzer
        model_name: gpt-4o
        system_prompt_path: analytica/analyzer_system   # OR
        system_prompt: "You are an analyst. ..."          # inline
        api_type: completion
        model_args:
            temperature: 0.1
            max_completion_tokens: 20000
        max_exception_retry: 3
        max_interrupt_steps: 5
        max_llm_recall: 0
        extra_settings: {}
    """

    name: str
    model: str
    system_prompt_path: Optional[str] = None
    system_prompt: Any = None          # Prompt object or None
    api_type: str = "completion"       # stored as string, converted at build time
    model_args: Dict[str, Any] = field(default_factory=dict)
    max_exception_retry: int = 3
    max_interrupt_steps: int = 5
    max_llm_recall: int = 0
    extra_settings: Dict[str, Any] = field(default_factory=dict)
    proxy: Optional[ProxyConfig] = None
    context_manager: Optional[ContextManagerConfig] = None

    @classmethod
    def from_config(cls, name: str, raw: Dict[str, Any]) -> "AgentSpec":
        """Parse a single agent config dict into an AgentSpec.

        *raw* is the per-agent dict **after** global defaults have been
        merged in.  Known keys are extracted; unknown keys are treated
        as additional model_args.
        """
        raw = raw.copy()

        # -- required: model -----------------------------------------------
        model = raw.pop("model_name", None)
        if model is None:
            raise ValueError(f"Agent '{name}' missing required 'model_name'")

        # -- required: system prompt (inline string or registry path) ------
        inline_prompt_str = raw.pop("system_prompt", None)
        system_prompt_path = raw.pop("system_prompt_path", None)
        if inline_prompt_str is None and system_prompt_path is None:
            raise ValueError(
                f"Agent '{name}' needs either 'system_prompt' or 'system_prompt_path'"
            )

        # Build a Prompt object from inline string if provided
        system_prompt = None
        if inline_prompt_str is not None:
            from lllm.core.prompt import Prompt
            system_prompt = Prompt(path=f"_inline/{name}/system", prompt=inline_prompt_str)

        # -- optional typed fields -----------------------------------------
        api_type = raw.pop("api_type", "completion")
        max_exception_retry = raw.pop("max_exception_retry", 3)
        max_interrupt_steps = raw.pop("max_interrupt_steps", 5)
        max_llm_recall = raw.pop("max_llm_recall", 0)
        extra_settings = raw.pop("extra_settings", {})

        # -- proxy config (already deep-merged with global by the caller) --
        proxy_raw = raw.pop("proxy", None)
        proxy = ProxyConfig.from_dict(proxy_raw) if proxy_raw else None

        # -- context manager config ----------------------------------------
        cm_raw = raw.pop("context_manager", None)
        if isinstance(cm_raw, dict):
            # type: null in YAML arrives as None value inside the dict
            cm_type = cm_raw.get("type")
            context_manager_cfg = None if cm_type is None else ContextManagerConfig.from_dict(cm_raw)
        else:
            context_manager_cfg = None

        # -- model_args: explicit dict + leftover unknown keys -------------
        model_args = raw.pop("model_args", {})
        raw.pop("name", None)
        if raw:
            logger.warning(
                "Agent '%s': unrecognised config keys %s will be passed as model_args. "
                "Known keys are: %s. Check for typos.",
                name,
                sorted(raw.keys()),
                sorted(_KNOWN_AGENT_KEYS),
            )
        model_args.update(raw)  # anything left is additional model_args

        return cls(
            name=name,
            model=model,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            api_type=api_type,
            model_args=model_args,
            max_exception_retry=max_exception_retry,
            max_interrupt_steps=max_interrupt_steps,
            max_llm_recall=max_llm_recall,
            extra_settings=extra_settings,
            proxy=proxy,
            context_manager=context_manager_cfg,
        )

    def build(self, runtime: Runtime, invoker):
        """Construct a live Agent from this spec."""
        from lllm.core.agent import Agent
        from lllm.core.const import APITypes

        if self.system_prompt is not None:
            prompt = self.system_prompt
        else:
            prompt = runtime.get_prompt(self.system_prompt_path)

        api_type = self.api_type if isinstance(self.api_type, APITypes) else APITypes(self.api_type)

        # -- Proxy tool injection ------------------------------------------
        if self.proxy is not None:
            from lllm.proxies.base import ProxyManager
            from lllm.proxies.interpreter import AgentInterpreter
            from lllm.proxies.proxy_tools import make_query_api_doc_tool, make_run_python_tool
            from lllm.proxies.prompt_template import render_proxy_prompt

            cutoff = (
                dt.datetime.fromisoformat(self.proxy.cutoff_date)
                if self.proxy.cutoff_date
                else None
            )
            proxy_manager = ProxyManager(
                activate_proxies=self.proxy.activate_proxies,
                cutoff_date=cutoff,
                deploy_mode=self.proxy.deploy_mode,
                runtime=runtime,
            )
            interpreter = AgentInterpreter(
                proxy_manager,
                max_output_chars=self.proxy.max_output_chars,
                truncation_indicator=self.proxy.truncation_indicator,
                timeout=self.proxy.timeout,
            )

            query_doc_tool = make_query_api_doc_tool(proxy_manager)

            # Only interpreter mode injects run_python.
            # Other modes (jupyter, None, future sandboxes) leave execution to
            # the tactic — the agent writes cell tags or uses another mechanism.
            extra_tools = [query_doc_tool]
            if self.proxy.exec_env == "interpreter":
                extra_tools.append(make_run_python_tool(interpreter))

            proxy_block = render_proxy_prompt(
                api_directory=proxy_manager.retrieve_api_docs(),
                max_output_chars=self.proxy.max_output_chars,
                truncation_indicator=self.proxy.truncation_indicator,
                exec_env=self.proxy.exec_env,
                custom_template=self.proxy.prompt_template,
            )

            # Create a modified prompt without mutating the original.
            # model_copy triggers model_post_init so _functions is rebuilt.
            prompt = prompt.model_copy(update={
                "prompt": prompt.prompt + proxy_block,
                "function_list": list(prompt.function_list) + extra_tools,
            })

        # -- Context manager -----------------------------------------------
        context_manager = (
            self.context_manager.build(self.model, runtime)
            if self.context_manager is not None
            else None
        )

        return Agent(
            name=self.name,
            system_prompt=prompt,
            model=self.model,
            llm_invoker=invoker,
            api_type=api_type,
            model_args=self.model_args,
            max_exception_retry=self.max_exception_retry,
            max_interrupt_steps=self.max_interrupt_steps,
            max_llm_recall=self.max_llm_recall,
            context_manager=context_manager,
        )


def parse_agent_configs(
    config: Dict[str, Any],
    agent_group: List[str],
    tactic_name: str,
) -> Dict[str, "AgentSpec"]:
    """Parse ``global`` + ``agent_configs`` from a tactic config dict.

    Returns ``{agent_name: AgentSpec}`` for each name in *agent_group*.
    """
    global_cfg = config.get("global", {})
    raw_list = config.get("agent_configs", [])

    agent_by_name: Dict[str, Dict] = {}
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise TypeError(f"agent_configs entries must be dicts, got {type(entry).__name__}")
        name = entry.get("name")
        if name is None:
            raise ValueError(f"Agent config entry missing 'name': {entry}")
        agent_by_name[name] = _deep_merge(global_cfg, entry)

    specs: Dict[str, AgentSpec] = {}
    for agent_name in agent_group:
        if agent_name not in agent_by_name:
            raise ValueError(
                f"Agent '{agent_name}' required by tactic '{tactic_name}' "
                f"not found in agent_configs. Available: {sorted(agent_by_name)}"
            )
        specs[agent_name] = AgentSpec.from_config(agent_name, agent_by_name[agent_name])

    return specs