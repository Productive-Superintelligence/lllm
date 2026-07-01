"""Tactic registration and construction helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from .tactic import _config_get

if TYPE_CHECKING:
    from .runtime import Registry, Runtime
    from .tactic import Tactic


def _get_default_registry() -> "Registry":
    from .runtime import get_default_registry

    return get_default_registry()


def _select_registry(
    *,
    runtime: Optional["Runtime"] = None,
    registry: Optional["Registry"] = None,
) -> "Registry":
    if runtime is not None and registry is not None and runtime is not registry:
        raise ValueError("Pass either runtime= or registry=, not both.")
    return registry or runtime or _get_default_registry()


def _normalize_name(name: Any) -> str:
    if isinstance(name, Enum) or (isinstance(name, type) and issubclass(name, Enum)):
        return name.value
    if isinstance(name, str):
        return name
    raise ValueError(f"Invalid tactic name: {name}")


def register_tactic_class(tactic_cls, runtime=None, *, registry=None):
    registry = _select_registry(runtime=runtime, registry=registry)
    name = _normalize_name(getattr(tactic_cls, "name", None))
    assert name not in (None, ""), (
        f"Tactic class {tactic_cls.__name__} must define `name`"
    )
    registry.register_tactic(name, tactic_cls, overwrite=True)
    return tactic_cls


def get_tactic_class(name, runtime=None, *, registry=None):
    registry = _select_registry(runtime=runtime, registry=registry)
    return registry.get_tactic(_normalize_name(name))


def _stable_tactic_id(namespace: str, tactic_name: str) -> str:
    """Return the stable physical identifier for a tactic: ``pkg::name``."""

    if namespace:
        package_name = namespace.split(".")[0]
        return f"{package_name}::{tactic_name}"
    return tactic_name


def build_tactic(
    config: Any,
    name: Optional[str] = None,
    runtime: Optional["Runtime"] = None,
    *,
    registry: Optional["Registry"] = None,
    **kwargs,
) -> "Tactic":
    """Build a Tactic from a config/profile dict."""

    if name is None:
        name = _config_get(config, "tactic_type")
    name = _normalize_name(name)
    rt = _select_registry(runtime=runtime, registry=registry)
    tactic_cls = get_tactic_class(name, registry=rt)

    try:
        node = rt.get_node(name, resource_type="tactic")
        tactic_path = _stable_tactic_id(node.namespace, tactic_cls.name)
    except (KeyError, AttributeError):
        tactic_path = tactic_cls.name

    return tactic_cls(
        config,
        registry=rt,
        tactic_path=tactic_path,
        **kwargs,
    )
