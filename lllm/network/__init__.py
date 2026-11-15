"""Deprecated: SSSN channels moved to the ``sssn`` package.

``lllm.network`` forwards to ``sssn`` so existing imports keep working, old
submodules included (``lllm.network.server`` is ``sssn.server``). New code
should ``import sssn``; install it with ``pip install sssn`` or
``pip install "lllm-core[network]"``.
"""

from __future__ import annotations

import importlib
import sys
import warnings

try:
    import sssn as _sssn
except ModuleNotFoundError as exc:
    if exc.name != "sssn":
        raise
    raise ImportError(
        "lllm.network moved to the sssn package: pip install sssn", name="sssn"
    ) from exc

warnings.warn(
    "lllm.network moved to the sssn package; use `import sssn` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_SUBMODULES = (
    "core",
    "stores",
    "server",
    "client",
    "resolver",
    "processing",
    "resources",
    "cli",
)

__all__ = list(getattr(_sssn, "__all__", ()))
globals().update({name: getattr(_sssn, name) for name in __all__})


def _alias_submodules() -> None:
    for name in _SUBMODULES:
        try:
            globals()[name] = importlib.import_module(f"sssn.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"sssn.{name}":
                raise
    # Alias every loaded sssn module, nested ones such as sssn.server.fastapi
    # included, so an old import path gives the same module object as sssn.
    for name, module in list(sys.modules.items()):
        if name.startswith("sssn."):
            sys.modules.setdefault(__name__ + name[len("sssn") :], module)


_alias_submodules()


def __getattr__(name: str):
    try:
        return getattr(_sssn, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
