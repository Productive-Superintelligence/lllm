"""
Auto-discovery of prompts and proxies from folders declared in ``lllm.toml``.

Every public function accepts an optional *context* parameter.  When omitted
the module falls back to :func:`~lllm.core.runtime.get_default_context`, so
callers that don't care about isolation never have to think about it.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import types
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from lllm.core.config import auto_discovery_disabled, load_config
from lllm.core.runtime import Context, get_default_context

if TYPE_CHECKING:
    from lllm.core.prompt import Prompt
    from lllm.proxies.base import BaseProxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORED_FILES = {"__init__.py", "__pycache__"}
PROMPT_SECTION = "prompts"
PROXY_SECTION = "proxies"

# Process-wide default; toggled via ``configure_auto_discover``.
_DEFAULT_AUTO_DISCOVER = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_discover(
    config_path: Optional[str | Path] = None,
    *,
    context: Optional[Context] = None,
    force: bool = False,
) -> None:
    """Scan ``lllm.toml`` folders and register every Prompt / BaseProxy found.

    Parameters
    ----------
    config_path:
        Explicit path to a ``lllm.toml`` (or directory containing one).
        Falls back to the normal resolution chain when *None*.
    context:
        The :class:`Context` to register into.  Defaults to the global context.
    force:
        Re-run discovery even if the context was already discovered.
    """
    ctx = context or get_default_context()

    if ctx._discovery_done and not force:
        return
    if auto_discovery_disabled():
        ctx._discovery_done = True
        return

    config = load_config(config_path)
    if not config:
        ctx._discovery_done = True
        return

    base_dir = Path(config["_config_path"]).parent
    try:
        _discover_prompts(config.get(PROMPT_SECTION, {}), base_dir, ctx)
        _discover_proxies(config.get(PROXY_SECTION, {}), base_dir, ctx)
    finally:
        ctx._discovery_done = True


def auto_discover_if_enabled(
    flag: Optional[bool] = None,
    config_path: Optional[str | Path] = None,
    *,
    context: Optional[Context] = None,
    force: bool = False,
) -> None:
    """Conditionally run :func:`auto_discover`.

    The *flag* parameter takes precedence.  When *None* the process-wide
    default set by :func:`configure_auto_discover` is used.
    """
    if not _should_auto_discover(flag):
        return
    auto_discover(config_path, context=context, force=force)


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


def _discover_prompts(section: dict, base_dir: Path, ctx: Context) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="prompts"):
            _register_prompts_from_module(module, namespace, ctx)


def _discover_proxies(section: dict, base_dir: Path, ctx: Context) -> None:
    folders = _normalize_paths(section.get("folders") or [], base_dir)
    for folder in folders:
        for module, namespace in _load_modules_from_folder(folder, prefix="proxies"):
            _register_proxies_from_module(module, namespace, ctx)


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
    module: types.ModuleType, namespace: str, ctx: Context
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
            ctx.register_prompt(attr, overwrite=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to register prompt '%s': %s", attr.path, exc)


def _register_proxies_from_module(
    module: types.ModuleType, namespace: str, ctx: Context
) -> None:
    from lllm.proxies.base import BaseProxy

    for attr_name, cls in vars(module).items():
        if not (inspect.isclass(cls) and issubclass(cls, BaseProxy) and cls is not BaseProxy):
            continue
        proxy_name = getattr(cls, "_proxy_path", f"{namespace}/{cls.__name__}")
        try:
            ctx.register_proxy(proxy_name, cls, overwrite=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to register proxy '%s': %s", proxy_name, exc)