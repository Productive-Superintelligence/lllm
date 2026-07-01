"""
Helpers for loading the stock proxies that ship with LLLM.

The actual proxy classes live in sibling modules (``exa_proxy``, ``fmp_proxy``...).
Importing those modules registers the proxies via ``ProxyRegistrator``. Projects that
do not leverage auto-discovery can call :func:`load_builtin_proxies` to register them
manually.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Iterable, List, Tuple

__all__ = ["load_builtin_proxies", "BUILTIN_PROXY_MODULES"]

BUILTIN_PROXY_MODULES: List[str] = [
    "lllm.runtimes.native.proxies.builtin.exa_proxy",
    "lllm.runtimes.native.proxies.builtin.fmp_proxy",
    "lllm.runtimes.native.proxies.builtin.fred_proxy",
    "lllm.runtimes.native.proxies.builtin.gt_proxy",
    "lllm.runtimes.native.proxies.builtin.msd_proxy",
    "lllm.runtimes.native.proxies.builtin.wa_proxy",
]


def load_builtin_proxies(modules: Iterable[str] | None = None) -> Tuple[List[str], Dict[str, Exception]]:
    """
    Import the packaged proxy modules so they register themselves.

    Args:
        modules: Optional iterable of module paths. Defaults to all bundled proxies.

    Returns:
        Tuple where the first element is a list of successfully imported modules and the
        second element is a mapping of ``module_path -> exception`` for imports that failed
        (missing optional dependencies, etc.). No exceptions are raised.
    """
    loaded: List[str] = []
    errors: Dict[str, Exception] = {}
    targets = list(modules or BUILTIN_PROXY_MODULES)
    for path in targets:
        try:
            import_module(path)
            loaded.append(path)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            errors[path] = exc
    return loaded, errors
