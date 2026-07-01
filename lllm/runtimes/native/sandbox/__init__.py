"""Optional native sandbox integrations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "JupyterSandbox": (".jupyter", "JupyterSandbox"),
    "JupyterSession": (".jupyter", "JupyterSession"),
    "JupyterCellType": (".jupyter", "JupyterCellType"),
    "ProgrammingLanguage": (".jupyter", "ProgrammingLanguage"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        try:
            module = import_module(module_name, __name__)
        except ModuleNotFoundError as exc:
            if exc.name in {"nbformat", "jupyter_client"}:
                raise RuntimeError(
                    "Native Jupyter sandbox support requires the sandbox extra. "
                    "Install it with `pip install 'lllm[sandbox]'`."
                ) from exc
            raise
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
