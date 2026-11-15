"""Deprecated: the AAAX strategy shell moved to the ``aaax`` package.

``lllm.shell`` forwards to ``aaax`` so existing imports keep working, old
submodules included (``lllm.shell.server`` is ``aaax.server``). New code
should ``import aaax``; install it with ``pip install aaax`` or
``pip install "lllm-core[shell]"``.
"""

from __future__ import annotations

import importlib
import sys
import warnings

try:
    import aaax as _aaax
except ModuleNotFoundError as exc:
    if exc.name != "aaax":
        raise
    raise ImportError(
        "lllm.shell moved to the aaax package: pip install aaax", name="aaax"
    ) from exc

warnings.warn(
    "lllm.shell moved to the aaax package; use `import aaax` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_SUBMODULES = ("cli", "loader", "models", "packages", "server", "strategy")

__all__ = list(getattr(_aaax, "__all__", ()))
globals().update({name: getattr(_aaax, name) for name in __all__})


def _alias_submodules() -> None:
    for name in _SUBMODULES:
        try:
            globals()[name] = importlib.import_module(f"aaax.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"aaax.{name}":
                raise
    # Alias every loaded aaax module so an old import path gives the same
    # module object as aaax.
    for name, module in list(sys.modules.items()):
        if name.startswith("aaax."):
            sys.modules.setdefault(__name__ + name[len("aaax") :], module)


_alias_submodules()


def __getattr__(name: str):
    try:
        return getattr(_aaax, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
