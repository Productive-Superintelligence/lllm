"""
Auto-discovery of prompts and proxies from folders declared in ``lllm.toml``.

Every public function accepts an optional *runtime* parameter.  When omitted
the module falls back to :func:`~lllm.core.runtime.get_default_runtime`, so
callers that don't care about isolation never have to think about it.
"""
from __future__ import annotations

import os
import importlib.util
import inspect
import logging
import types
import warnings
from pathlib import Path
from typing import Iterable, Optional, List, Dict
import tomllib  

from lllm.core.runtime import Runtime, get_default_runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORED_FILES = {"__init__.py", "__pycache__"}
PROMPT_SECTION = "prompts"
PROXY_SECTION = "proxies"

# Process-wide default; toggled via ``configure_auto_discover``.
_DEFAULT_AUTO_DISCOVER = True



LLLM_CONFIG_ENV = "LLLM_CONFIG"
LLLM_CONFIG_DISABLE_ENV = "LLLM_AUTO_DISCOVER"
CONFIG_FILENAMES = ("lllm.toml", ".lllm.toml", "LLLM.toml")
CONFIG_SUBDIRS = ("", "template")


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


def _resolve_candidate(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "lllm.toml"
    if candidate.is_file():
        return candidate.resolve()
    return None


def find_config_file(start_path: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
    """
    Locate the nearest lllm.toml by checking:
      1. The LLLM_CONFIG environment variable (file or directory)
      2. The provided start_path and its parents
      3. The current working directory and its parents
    """
    env_path = _resolve_candidate(os.environ.get(LLLM_CONFIG_ENV))
    if env_path:
        return env_path

    search_roots: List[Path] = []
    if start_path is not None:
        search_roots.append(Path(start_path).resolve())
    search_roots.append(Path.cwd())

    for root in search_roots:
        for path in [root, *root.parents]:
            for subdir in CONFIG_SUBDIRS:
                base = path if subdir == "" else path / subdir
                for name in CONFIG_FILENAMES:
                    candidate = base / name
                    if candidate.is_file():
                        return candidate.resolve()
    return None


def load_config(path: Optional[str | os.PathLike[str]] = None) -> Optional[Dict]:
    config_path = _resolve_candidate(path) or find_config_file(path)
    if not config_path:
        return None
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    data["_config_path"] = config_path
    return data


def auto_discovery_disabled() -> bool:
    return os.environ.get(LLLM_CONFIG_DISABLE_ENV, "1").lower() in {"0", "false", "no"}



# ---------------------------------------------------------------------------
# Auto Discovery 
# ---------------------------------------------------------------------------

def auto_discover(
    config_path: Optional[str | Path] = None,
    *,
    runtime: Optional[Runtime] = None,
    force: bool = False,
) -> None:
    """Scan ``lllm.toml`` folders and register every Prompt / BaseProxy found.

    Parameters
    ----------
    config_path:
        Explicit path to a ``lllm.toml`` (or directory containing one).
        Falls back to the normal resolution chain when *None*.
    runtime:
        The :class:`Runtime` to register into.  Defaults to the global runtime.
    force:
        Re-run discovery even if the runtime was already discovered.
    """
    runtime = runtime or get_default_runtime()

    if runtime._discovery_done and not force:
        return
    if auto_discovery_disabled():
        runtime._discovery_done = True
        return

    config = load_config(config_path)
    if not config:
        runtime._discovery_done = True
        return

    base_dir = Path(config["_config_path"]).parent
    try:
        _discover_prompts(config.get(PROMPT_SECTION, {}), base_dir, runtime)
        _discover_proxies(config.get(PROXY_SECTION, {}), base_dir, runtime)
    finally:
        runtime._discovery_done = True


def auto_discover_if_enabled(
    flag: Optional[bool] = None,
    config_path: Optional[str | Path] = None,
    *,
    runtime: Optional[Runtime] = None,
    force: bool = False,
) -> None:
    """Conditionally run :func:`auto_discover`.

    The *flag* parameter takes precedence.  When *None* the process-wide
    default set by :func:`configure_auto_discover` is used.
    """
    if not _should_auto_discover(flag):
        return
    auto_discover(config_path, runtime=runtime, force=force)


def configure_auto_discover(enabled: bool) -> None:
    """Set the process-wide default for ``auto_discover_if_enabled``."""
    global _DEFAULT_AUTO_DISCOVER
    _DEFAULT_AUTO_DISCOVER = bool(enabled)


# ---------------------------------------------------------------------------
# Internal helpers — discovery logic
# ---------------------------------------------------------------------------

def _should_auto_discover(flag: Optional[bool]) -> bool:
    """Resolve an explicit flag against the process-wide default."""
    if flag is None:
        return _DEFAULT_AUTO_DISCOVER
    return bool(flag)


def _discover_prompts(section: dict, base_dir: Path, runtime: Runtime) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="prompts"):
            _register_prompts_from_module(module, namespace, runtime)


def _discover_proxies(section: dict, base_dir: Path, runtime: Runtime) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="proxies"):
            _register_proxies_from_module(module, namespace, runtime)


# ---------------------------------------------------------------------------
# Internal helpers — filesystem & module loading
# ---------------------------------------------------------------------------

def _normalize_paths(entries: Iterable[str], base_dir: Path) -> list[Path]:
    """Resolve folder entries relative to *base_dir*, skip missing ones."""
    normalized: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.exists():
            normalized.append(path)
        else:
            warnings.warn(
                f"LLLM discovery skipped missing path: {path}",
                RuntimeWarning,
                stacklevel=3,
            )
    return normalized


def _load_modules_from_folder(
    folder: Path, prefix: str
) -> Iterable[tuple[types.ModuleType, str]]:
    """Import every ``*.py`` in *folder* (non-recursive), yielding (module, namespace)."""
    for file in sorted(folder.glob("*.py")):
        if file.name in IGNORED_FILES or file.name.startswith("_"):
            continue
        namespace = f"{prefix}.{folder.name}.{file.stem}"
        try:
            module = _load_module_from_file(file, namespace)
        except Exception as exc:  # pragma: no cover — best-effort discovery
            warnings.warn(
                f"LLLM discovery failed to load {file}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        yield module, file.stem


def _load_module_from_file(file_path: Path, namespace: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(namespace, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Internal helpers — registration
# ---------------------------------------------------------------------------

def _register_prompts_from_module(
    module: types.ModuleType, namespace: str, runtime: Runtime
) -> None:
    # Import Prompt here to avoid circular imports at module level
    from lllm.core.prompt import Prompt

    for attr_name, attr in vars(module).items():
        if not isinstance(attr, Prompt):
            continue
        # Auto-prefix the path when the prompt doesn't already have a namespace
        if "/" not in attr.path:
            attr.path = f"{namespace}/{attr.path}".strip("/")
        try:
            runtime.register_prompt(attr, overwrite=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to register prompt '%s': %s", attr.path, exc)


def _register_proxies_from_module(
    module: types.ModuleType, namespace: str, runtime: Runtime
) -> None:
    from lllm.proxies.base import BaseProxy

    for attr_name, cls in vars(module).items():
        if not (inspect.isclass(cls) and issubclass(cls, BaseProxy) and cls is not BaseProxy):
            continue
        proxy_name = getattr(cls, "_proxy_path", f"{namespace}/{cls.__name__}")
        try:
            runtime.register_proxy(proxy_name, cls, overwrite=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to register proxy '%s': %s", proxy_name, exc)