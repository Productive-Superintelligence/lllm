# lllm/core/resource.py
"""
Resource registry primitives.

``ResourceNode`` wraps anything stored in a Runtime registry.
``PackageInfo`` captures the identity of a loaded LLLM package.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from lllm.core.prompt import Prompt
    from lllm.core.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass
class PackageInfo:
    """Metadata for one loaded LLLM package."""
    name: str
    version: str = ""
    description: str = ""
    base_dir: str = ""
    alias: Optional[str] = None

    @property
    def effective_name(self) -> str:
        return self.alias or self.name


@dataclass
class ResourceNode:
    """
    Universal registry entry.  Wraps any resource with namespace
    qualification and optional lazy loading.

    Access the wrapped object via ``.value`` — triggers the loader on
    first access if one was provided.
    """
    key: str
    namespace: str = ""
    resource_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    _value: Any = field(default=None, repr=False)
    _loader: Optional[Callable[[], Any]] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    @property
    def qualified_key(self) -> str:
        if self.namespace:
            return f"{self.namespace}:{self.key}"
        return self.key

    @property
    def value(self) -> Any:
        if not self._loaded:
            if self._loader is not None:
                try:
                    self._value = self._loader()
                except Exception as exc:
                    logger.error("Failed to load resource '%s': %s",
                                 self.qualified_key, exc)
                    raise
            self._loaded = True
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        self._value = v
        self._loaded = True
        self._loader = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @classmethod
    def eager(cls, key, value, namespace="", resource_type="generic", **meta):
        return cls(key=key, namespace=namespace, resource_type=resource_type,
                   metadata=meta, _value=value, _loaded=True)

    @classmethod
    def lazy(cls, key, loader, namespace="", resource_type="generic", **meta):
        return cls(key=key, namespace=namespace, resource_type=resource_type,
                   metadata=meta, _loader=loader, _loaded=False)

    def __repr__(self):
        tag = "loaded" if self._loaded else "lazy"
        return f"ResourceNode({self.qualified_key!r}, type={self.resource_type!r}, {tag})"



# ---------------------------------------------------------------------------
# Public convenience loaders
# ---------------------------------------------------------------------------

def load_prompt(path: str, runtime: "Optional[Runtime]" = None) -> "Prompt":
    """Load a prompt.  Accepts ``"resource"``, ``"pkg:resource"``,
    or ``"pkg.prompts:resource"``."""
    from lllm.core.runtime import get_default_runtime
    return (runtime or get_default_runtime()).get_prompt(path)


def load_tactic(path: str, runtime: "Optional[Runtime]" = None):
    """Load a tactic class."""
    from lllm.core.runtime import get_default_runtime
    return (runtime or get_default_runtime()).get_tactic(path)


def load_proxy(path: str, runtime: "Optional[Runtime]" = None):
    """Load a proxy class."""
    from lllm.core.runtime import get_default_runtime
    return (runtime or get_default_runtime()).get_proxy(path)


def load_config(path: str, runtime: "Optional[Runtime]" = None) -> Any:
    """Load a config dict (triggers lazy file read if needed)."""
    from lllm.core.runtime import get_default_runtime
    return (runtime or get_default_runtime()).get_config(path)


def load_resource(path: str, runtime: "Optional[Runtime]" = None) -> Any:
    """Load any resource by full URL.  Requires ``"pkg.section:resource"``
    or ``"section:resource"`` (section-only resolves via default package).

    Raises ``ValueError`` if no ``:`` in path.
    """
    from lllm.core.runtime import get_default_runtime

    if ":" not in path:
        raise ValueError(
            f"load_resource requires '<package.section>:<resource>' format, "
            f"got '{path}'. Use load_prompt/load_tactic/etc. for bare paths."
        )

    rt = runtime or get_default_runtime()
    pkg_part, resource_part = path.split(":", 1)

    if "." not in pkg_part and rt.default_namespace:
        full_key = f"{rt.default_namespace}.{pkg_part}:{resource_part}"
    else:
        full_key = path

    return rt.get(full_key)
