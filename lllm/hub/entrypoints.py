"""Entrypoint import helpers for validating and serving local LLLM hub packages."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def import_entrypoint(
    entry: str,
    *,
    base_dir: Path | None = None,
    keep_loaded: bool = False,
) -> Any:
    """Import ``module:attribute``, resolving the module from ``base_dir``.

    When the package provides the entry's top-level module, modules imported
    under that name from elsewhere are set aside first, so the entry comes
    from the package. By default the import is isolated, as validation needs:
    afterwards ``base_dir`` leaves ``sys.path`` and the set-aside modules are
    restored.

    With ``keep_loaded``, as serving needs, the package's modules and its
    ``sys.path`` entry stay for the life of the process, so package code can
    import its own modules when it is called. Packages served in one process
    share ``sys.modules``, so call-time imports need distinct top-level module
    names: loading a package whose top-level module has the same name as an
    already served package's replaces that package's modules.
    """

    if (
        not isinstance(entry, str)
        or not entry
        or entry != entry.strip()
    ):
        raise ValueError(f"Entrypoint must have shape module:attribute: {entry}")
    module_name, sep, attr_path = entry.partition(":")
    if (
        not sep
        or not _entrypoint_segments(module_name)
        or not _entrypoint_segments(attr_path)
    ):
        raise ValueError(f"Entrypoint must have shape module:attribute: {entry}")
    root_module = module_name.split(".", 1)[0]
    with _import_path(base_dir, root_module=root_module, keep_loaded=keep_loaded):
        module = importlib.import_module(module_name)
        value: Any = module
        for part in attr_path.split("."):
            value = getattr(value, part)
        return value


def _entrypoint_segments(value: str) -> bool:
    return all(
        part
        and not any(ch.isspace() for ch in part)
        and not any(ch in part for ch in "/\\:%")
        for part in value.split(".")
    )


@contextmanager
def _import_path(
    base_dir: Path | None,
    *,
    root_module: str | None = None,
    keep_loaded: bool = False,
) -> Iterator[None]:
    if base_dir is None:
        yield
        return
    value = str(base_dir)
    isolated_modules: dict[str, Any] = {}
    isolate = root_module is not None and (
        (base_dir / root_module).exists() or (base_dir / f"{root_module}.py").exists()
    )
    # A package loaded to serve keeps one copy of its modules: loading another
    # of its entries reuses them instead of importing the package again.
    if isolate and not (keep_loaded and _loaded_from(base_dir, root_module)):
        for name in list(sys.modules):
            if name == root_module or name.startswith(f"{root_module}."):
                isolated_modules[name] = sys.modules.pop(name)
    if keep_loaded and value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
    importlib.invalidate_caches()
    if keep_loaded:
        yield
        return
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
        if isolate:
            for name in list(sys.modules):
                if name == root_module or name.startswith(f"{root_module}."):
                    sys.modules.pop(name, None)
            sys.modules.update(isolated_modules)
        importlib.invalidate_caches()


def _loaded_from(base_dir: Path, root_module: str) -> bool:
    """Whether ``root_module`` is already imported from ``base_dir``."""

    origin = getattr(sys.modules.get(root_module), "__file__", None)
    return origin is not None and Path(origin).parent in (base_dir, base_dir / root_module)
