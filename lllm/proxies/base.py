import inspect
import functools as ft
import datetime as dt
from typing import Dict, Any, List, Optional, Callable
import lllm.utils as U
from lllm.core.context import Context, get_default_context


class BaseProxy:
    """Base class for describing an API surface that agents can call as tools."""

    def __init__(
        self,
        *args,
        activate_proxies: Optional[List[str]] = None,
        cutoff_date: Optional[dt.datetime] = None,
        deploy_mode: bool = False,
        use_cache: bool = True,
        auto_discover: Optional[bool] = None,
        **kwargs,
    ):
        self.activate_proxies = activate_proxies[:] if activate_proxies else []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self.use_cache = use_cache
        self.auto_discover = auto_discover

        if isinstance(self.cutoff_date, str):
            try:
                self.cutoff_date = dt.datetime.fromisoformat(self.cutoff_date)
            except ValueError:
                self.cutoff_date = None

    @staticmethod
    def endpoint(category: str, endpoint: str, description: str, params: dict, response: list,
                 name: str = None, sub_category: str = None, remove_keys: list = None,
                 dt_cutoff: tuple = None, method: str = 'GET'):
        """Decorator that records metadata about an API endpoint."""
        def decorator(func):
            func.endpoint_info = {
                'category': category,
                'endpoint': endpoint,
                'name': name,
                'description': description,
                'sub_category': sub_category,
                'remove_keys': remove_keys,
                'params': params,
                'response': response,
                'dt_cutoff': dt_cutoff,
                'method': method
            }
            return func
        return decorator

    @staticmethod
    def postcall(func):
        func.is_postcall = True
        return func

    # ------------------------------------------------------------------
    # Endpoint metadata helpers
    # ------------------------------------------------------------------

    def _endpoint_methods(self):
        """
        Yield ``(attr_name, method, endpoint_info)`` triples for every method
        decorated with :func:`BaseProxy.endpoint`.
        """
        for name, method in inspect.getmembers(self, predicate=callable):
            info = getattr(method, "endpoint_info", None)
            if info:
                yield name, method, info

    def endpoint_directory(self) -> List[Dict[str, Any]]:
        """
        Return a structured list describing every endpoint exposed by this proxy.
        """
        directory: List[Dict[str, Any]] = []
        for name, method, info in self._endpoint_methods():
            entry = dict(info)
            entry.setdefault("name", info.get("name") or name)
            entry["callable"] = name
            entry["docstring"] = inspect.getdoc(method)
            directory.append(entry)
        directory.sort(key=lambda item: ((item.get("category") or ""), item.get("endpoint") or ""))
        return directory

    def api_directory(self) -> Dict[str, Any]:
        """
        Return proxy metadata plus the endpoint directory.
        """
        return {
            "id": getattr(self, "_proxy_path", self.__class__.__name__),
            "display_name": getattr(self, "_proxy_name", self.__class__.__name__),
            "description": getattr(self, "_proxy_description", ""),
            "endpoints": self.endpoint_directory(),
        }

    def auto_test(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform light-weight validation of endpoint metadata.

        Returns a dict mapping callable name to ``{"status": "...", "issues": [...]}``.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for entry in self.endpoint_directory():
            issues: List[str] = []
            params = entry.get("params")
            if not isinstance(params, dict):
                issues.append("params")
            if entry.get("response") is None:
                issues.append("response")
            status = "ok" if not issues else "warning"
            results[entry["callable"]] = {"status": status, "issues": issues}
        return results

class Proxy:
    """
    Runtime registry that instantiates every discovered proxy and forwards calls.

    Agents rarely instantiate this directly; instead the sandbox or higher-level tooling
    wires it up so prompts can enumerate available endpoints for tool selection.
    """

    def __init__(
        self,
        activate_proxies: Optional[List[str]] = None,
        cutoff_date: dt.datetime = None,
        deploy_mode: bool = False,
        context: Context = None,
        *,
        auto_discover: Optional[bool] = None,
    ):
        from lllm.core.discovery import auto_discover_if_enabled
        auto_discover_if_enabled(auto_discover)
        self.activate_proxies = activate_proxies or []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self._context = context or get_default_context()
        self._auto_discover_flag = auto_discover
        self._load_registered_proxies()

    def _load_registered_proxies(self):
        for name, proxy_cls in self._context.proxies.items():
            if self.activate_proxies and name not in self.activate_proxies:
                continue
            instance = proxy_cls(
                cutoff_date=self.cutoff_date,
                activate_proxies=self.activate_proxies,
                deploy_mode=self.deploy_mode,
                auto_discover=self._auto_discover_flag,
            )
            self.proxies[name] = instance

    def register(self, name: str, proxy_cls: Any):
        """Register (or override) a proxy implementation at runtime."""
        if name in self.proxies:
            U.cprint(f'Proxy {name} already instantiated, overwriting instance', 'y')
        try:
            instance = proxy_cls(
                cutoff_date=self.cutoff_date,
                activate_proxies=self.activate_proxies,
                deploy_mode=self.deploy_mode,
                auto_discover=self._auto_discover_flag,
            )
        except TypeError:
            instance = proxy_cls(self.activate_proxies, self.cutoff_date, self.deploy_mode)
        self.proxies[name] = instance

    def available(self) -> List[str]:
        """Return the sorted list of proxy identifiers currently loaded."""
        return sorted(self.proxies.keys())

    def api_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Return the API directory for every loaded proxy."""
        return {name: proxy.api_directory() for name, proxy in self.proxies.items()}

    def get_api_directory(self, proxy_name: str) -> Dict[str, Any]:
        """Convenience wrapper that returns the directory for a single proxy."""
        if proxy_name not in self.proxies:
            raise KeyError(f"Proxy '{proxy_name}' not registered")
        return self.proxies[proxy_name].api_directory()

    def retrieve_api_docs(self, proxy_name: Optional[str] = None) -> str:
        """
        Render a human-readable overview of the available endpoints.

        Args:
            proxy_name: Optional identifier. If omitted, all proxies are included.
        """
        if proxy_name is not None:
            target_names = [proxy_name]
        else:
            target_names = sorted(self.proxies.keys())

        sections: List[str] = []
        for name in target_names:
            if name not in self.proxies:
                raise KeyError(f"Proxy '{name}' not registered")
            meta = self.proxies[name].api_directory()
            header = f"## {meta['display_name']} ({name})"
            description = (meta.get("description") or "").strip()
            lines = [header]
            if description:
                lines.append(description)
            for endpoint in meta.get("endpoints", []):
                method = (endpoint.get("method") or "GET").upper()
                desc = endpoint.get("description") or ""
                line = f"- **{endpoint.get('endpoint')}** [{method}] – {desc}"
                lines.append(line.strip())
                params = endpoint.get("params") or {}
                if isinstance(params, dict) and params:
                    for param_name, spec in params.items():
                        if isinstance(spec, tuple) and spec:
                            type_hint = spec[0]
                            example = spec[1] if len(spec) > 1 else None
                        else:
                            type_hint = None
                            example = spec
                        type_name = getattr(type_hint, "__name__", str(type_hint)) if type_hint else "Any"
                        if isinstance(example, (list, dict)):
                            example_preview = str(example)[:60]
                        else:
                            example_preview = example
                        lines.append(f"    - `{param_name}` ({type_name}) e.g. {example_preview}")
            sections.append("\n".join(lines).strip())
        return "\n\n".join(sections).strip()

    def _resolve(self, endpoint: str) -> tuple[str, str]:
        if '.' in endpoint:
            parts = endpoint.split('.', 1)
            return parts[0], parts[1]
        path_parts = endpoint.split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid endpoint '{endpoint}'. Use '<proxy>.<method>' or '<proxy>/<method>'.")
        return '/'.join(path_parts[:-1]), path_parts[-1]

    def __call__(self, endpoint: str, *args, **kwargs):
        """Dispatch ``proxy_path.endpoint_name`` or ``proxy_path/endpoint`` to the proxy."""
        proxy_name, func_name = self._resolve(endpoint)
        if proxy_name not in self.proxies:
            raise KeyError(f"Proxy '{proxy_name}' not registered. Available: {list(self.proxies.keys())}")
        proxy = self.proxies[proxy_name]
        if not hasattr(proxy, func_name):
            raise AttributeError(f"Proxy '{proxy_name}' has no endpoint '{func_name}'")
        handler = getattr(proxy, func_name)
        return handler(*args, **kwargs)

def ProxyRegistrator(path: str, name: str, description: str, context: Context = None):
    ctx = context or get_default_context()
    def decorator(cls):
        cls._proxy_path = path
        cls._proxy_name = name
        cls._proxy_description = description
        ctx.register_proxy(path, cls, overwrite=True)
        return cls
    return decorator

def register_proxy(name: str, proxy_cls, overwrite: bool = False):
    get_default_context().register_proxy(name, proxy_cls, overwrite)