# lllm/core/runtime.py
"""
Runtime — the central registry for an LLLM session.

All resources live in a unified ``ResourceNode``-based store keyed by
qualified URLs (``"package.section:resource_path"``).  Resolution is
namespace-aware: bare keys are resolved via the default namespace.

Named runtimes are supported for parallel experiments.
"""
from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING

from lllm.core.resource import ResourceNode, PackageInfo

if TYPE_CHECKING:
    from lllm.core.prompt import Prompt
    from lllm.proxies.base import BaseProxy

logger = logging.getLogger(__name__)


class Runtime:
    """
    Holds all registries and shared state for an LLLM runtime.

    Internally every resource is a :class:`ResourceNode` stored in
    ``_resources`` keyed by its qualified key.  Typed convenience
    methods (``register_prompt``, ``get_prompt``, etc.) are thin wrappers.
    """

    def __init__(self) -> None:
        self._resources: Dict[str, ResourceNode] = {}
        self._type_index: Dict[str, Set[str]] = {}
        self.packages: Dict[str, PackageInfo] = {}
        self._loaded_package_paths: Set[str] = set()
        self._default_namespace: Optional[str] = None
        self._discovery_done: bool = False

    # ==================================================================
    # Unified registry
    # ==================================================================

    def register(self, node: ResourceNode, overwrite: bool = True) -> None:
        """Register a ``ResourceNode``.

        Stored under ``node.qualified_key``.  Warns on collision if
        overwriting.
        """
        qk = node.qualified_key

        if qk in self._resources:
            if not overwrite:
                raise ValueError(f"Resource '{qk}' already registered")
            existing = self._resources[qk]
            if existing is not node:
                logger.debug("Resource '%s' overwritten (type=%s)", qk, node.resource_type)

        self._resources[qk] = node

        rtype = node.resource_type
        if rtype not in self._type_index:
            self._type_index[rtype] = set()
        self._type_index[rtype].add(qk)

    def get(self, key: str, resource_type: Optional[str] = None) -> Any:
        """Retrieve a resource value by key.

        Resolution:
            1. Exact match on *key*.
            2. If *key* has no ``:``, try ``default_ns.<type>s:key`` for
               the given *resource_type*, or scan all built-in sections.
        """
        node = self._resolve(key, resource_type)
        if resource_type and node.resource_type != resource_type:
            raise TypeError(
                f"Resource '{key}' is type '{node.resource_type}', expected '{resource_type}'"
            )
        return node.value

    def get_node(self, key: str, resource_type: Optional[str] = None) -> ResourceNode:
        """Like :meth:`get` but returns the node itself (not ``.value``)."""
        return self._resolve(key, resource_type)

    def _resolve(self, key: str, resource_type: Optional[str] = None) -> ResourceNode:
        # 1. Exact match
        if key in self._resources:
            return self._resources[key]

        # 2. Default namespace fallback (bare key, no ":")
        if self._default_namespace and ":" not in key:
            if resource_type:
                ns_key = f"{self._default_namespace}.{resource_type}s:{key}"
                if ns_key in self._resources:
                    return self._resources[ns_key]
            for section in ("prompts", "proxies", "tactics", "configs"):
                ns_key = f"{self._default_namespace}.{section}:{key}"
                if ns_key in self._resources:
                    return self._resources[ns_key]

        # 3. If "pkg:path" with no dot, inject section for typed lookups
        if ":" in key and resource_type:
            pkg_part, resource_part = key.split(":", 1)
            if "." not in pkg_part:
                full_key = f"{pkg_part}.{resource_type}s:{resource_part}"
                if full_key in self._resources:
                    return self._resources[full_key]

        all_keys = sorted(self._resources.keys())
        close = difflib.get_close_matches(key, all_keys, n=5, cutoff=0.4)
        hint = f" Did you mean: {close}?" if close else f" Registered ({len(all_keys)}): {all_keys[:20]}"
        raise KeyError(f"Resource '{key}' not found.{hint}")

    def has(self, key: str) -> bool:
        try:
            self._resolve(key)
            return True
        except KeyError:
            return False

    def keys(self, resource_type: Optional[str] = None) -> List[str]:
        if resource_type:
            return sorted(self._type_index.get(resource_type, set()))
        return sorted(self._resources.keys())

    # ==================================================================
    # Package management
    # ==================================================================

    def register_package(self, pkg: PackageInfo) -> None:
        eff = pkg.effective_name
        if eff in self.packages:
            existing = self.packages[eff]
            if existing.base_dir == pkg.base_dir:
                return
            logger.warning(
                "Package '%s' already registered (from %s), overwriting with %s",
                eff, existing.base_dir, pkg.base_dir,
            )
        self.packages[eff] = pkg

    # ==================================================================
    # Typed convenience — Prompts
    # ==================================================================

    def register_prompt(self, prompt: "Prompt", overwrite: bool = True,
                        namespace: str = "") -> None:
        node = ResourceNode.eager(prompt.path, prompt,
                                  namespace=namespace, resource_type="prompt")
        self.register(node, overwrite=overwrite)
        prompt._qualified_key = node.qualified_key  # type: ignore[attr-defined]

    def get_prompt(self, path: str) -> "Prompt":
        return self.get(path, resource_type="prompt")

    # ==================================================================
    # Typed convenience — Proxies
    # ==================================================================

    def register_proxy(self, name: str, proxy_cls: Type["BaseProxy"],
                       overwrite: bool = False, namespace: str = "") -> None:
        node = ResourceNode.eager(name, proxy_cls,
                                  namespace=namespace, resource_type="proxy")
        self.register(node, overwrite=overwrite)

    def get_proxy(self, path: str) -> Type["BaseProxy"]:
        return self.get(path, resource_type="proxy")

    # ==================================================================
    # Typed convenience — Tactics
    # ==================================================================

    def register_tactic(self, tactic_type: str, tactic_cls: Type,
                        overwrite: bool = False, namespace: str = "") -> None:
        node = ResourceNode.eager(tactic_type, tactic_cls,
                                  namespace=namespace, resource_type="tactic")
        self.register(node, overwrite=overwrite)

    def get_tactic(self, name: str) -> Type:
        return self.get(name, resource_type="tactic")

    # ==================================================================
    # Typed convenience — Configs
    # ==================================================================

    def register_config(self, name: str, config_data: Any = None,
                        overwrite: bool = True, namespace: str = "",
                        loader: Any = None) -> None:
        if loader is not None:
            node = ResourceNode.lazy(name, loader,
                                     namespace=namespace, resource_type="config")
        else:
            node = ResourceNode.eager(name, config_data,
                                      namespace=namespace, resource_type="config")
        self.register(node, overwrite=overwrite)

    def get_config(self, path: str) -> Any:
        return self.get(path, resource_type="config")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def reset(self) -> None:
        self._resources.clear()
        self._type_index.clear()
        self.packages.clear()
        self._loaded_package_paths.clear()
        self._default_namespace = None
        self._discovery_done = False


# ==================================================================
# Module-level singletons and named runtime registry
# ==================================================================

_default_runtime = Runtime()
_runtimes: Dict[str, Runtime] = {}


def get_default_runtime() -> Runtime:
    return _default_runtime


def set_default_runtime(rt: Runtime) -> None:
    global _default_runtime
    _default_runtime = rt


def get_runtime(name: Optional[str] = None) -> Runtime:
    if name is None:
        return _default_runtime
    if name not in _runtimes:
        raise KeyError(
            f"Runtime '{name}' not found. Available: {sorted(_runtimes)}. "
            f"Call load_runtime(path, name='{name}') first."
        )
    return _runtimes[name]


def load_runtime(
    toml_path: Optional[str] = None,
    name: Optional[str] = None,
) -> Runtime:
    """Create and populate a Runtime from a TOML file.

    *name=None* → replaces the default runtime.
    *name="something"* → stored as a named runtime.
    """
    from lllm.core.config import load_package, find_config_file

    rt = Runtime()
    if toml_path is not None:
        load_package(str(toml_path), runtime=rt)
    else:
        found = find_config_file()
        if found:
            load_package(str(found), runtime=rt)
        else:
            logger.info(
                "No lllm.toml found. Running with empty runtime (fast mode)."
            )

    rt._discovery_done = True

    if name is None:
        set_default_runtime(rt)
    else:
        _runtimes[name] = rt
    return rt