import datetime as dt
import inspect
import logging
from typing import Any, Dict, List, Optional

from lllm.core.runtime import Runtime, get_default_runtime

logger = logging.getLogger(__name__)


class BaseProxy:
    """Base class for describing an API surface that agents can call as tools."""

    def __init__(
        self,
        *args,
        activate_proxies: Optional[List[str]] = None,
        cutoff_date: Optional[dt.datetime] = None,
        deploy_mode: bool = False,
        use_cache: bool = True,
        runtime: Optional[Runtime] = None,
        **kwargs,
    ):
        self._runtime = runtime or get_default_runtime()
        self._qualified_key = getattr(self.__class__, "_qualified_key", None)
        self._resource_namespace = getattr(self.__class__, "_resource_namespace", None)
        self.activate_proxies = activate_proxies[:] if activate_proxies else []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self.use_cache = use_cache

        if isinstance(self.cutoff_date, str):
            try:
                self.cutoff_date = dt.datetime.fromisoformat(self.cutoff_date)
            except ValueError:
                self.cutoff_date = None

    @staticmethod
    def endpoint(
        category: str,
        endpoint: str,
        description: str,
        params: dict,
        response: list,
        name: Optional[str] = None,
        sub_category: Optional[str] = None,
        remove_keys: Optional[list] = None,
        dt_cutoff: Optional[tuple] = None,
        method: str = "GET",
    ):
        """Decorator that records metadata about an API endpoint."""

        def decorator(func):
            func.endpoint_info = {
                "category": category,
                "endpoint": endpoint,
                "name": name,
                "description": description,
                "sub_category": sub_category,
                "remove_keys": remove_keys,
                "params": params,
                "response": response,
                "dt_cutoff": dt_cutoff,
                "method": method,
            }
            return func

        return decorator

    @staticmethod
    def postcall(func):
        func.is_postcall = True
        return func

    @staticmethod
    def tactic_endpoint(
        tactic_url: str,
        *,
        config: Any = None,
        endpoint: Optional[str] = None,
        category: str = "tactics",
        name: Optional[str] = None,
        description: Optional[str] = None,
        response: Any = None,
        method: str = "TACTIC",
    ):
        """Create a proxy endpoint backed by a tactic resource URL."""
        from lllm.core.tactic_tool import sanitize_tool_name

        endpoint_name = endpoint or sanitize_tool_name(tactic_url)

        def _call_tactic_endpoint(self, params: Any = None, **kwargs):
            from lllm.core.tactic_tool import call_tactic_proxy_endpoint

            return call_tactic_proxy_endpoint(
                tactic_url,
                params=params,
                kwargs=kwargs,
                runtime=getattr(self, "_runtime", None),
                base_namespace=getattr(self, "_resource_namespace", None),
                config=config,
            )

        _call_tactic_endpoint.__name__ = endpoint_name
        _call_tactic_endpoint.endpoint_info = {
            "category": category,
            "endpoint": endpoint_name,
            "name": name or endpoint_name,
            "description": description or f"Run tactic {tactic_url}.",
            "sub_category": None,
            "remove_keys": None,
            "params": {},
            "response": response if response is not None else "str",
            "dt_cutoff": None,
            "method": method,
        }
        _call_tactic_endpoint._lllm_tactic_endpoint = {
            "tactic_url": tactic_url,
            "config": config,
            "endpoint": endpoint,
            "category": category,
            "name": name,
            "description": description,
            "response": response,
            "method": method,
        }
        return _call_tactic_endpoint

    @classmethod
    def register_tactic(
        cls,
        tactic_url: str,
        *,
        endpoint: str = None,
        config: Any = None,
        category: str = "tactics",
        name: str = None,
        description: str = None,
        response: Any = None,
        method: str = "TACTIC",
    ):
        """Imperatively add a tactic-backed endpoint to this proxy class."""
        from lllm.core.tactic_tool import sanitize_tool_name

        attr_name = sanitize_tool_name(endpoint or name or tactic_url)
        method_fn = BaseProxy.tactic_endpoint(
            tactic_url,
            config=config,
            endpoint=endpoint or attr_name,
            category=category,
            name=name,
            description=description,
            response=response,
            method=method,
        )
        setattr(cls, attr_name, method_fn)
        return method_fn

    # ------------------------------------------------------------------
    # Endpoint metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _callable_attr(method, attr: str, default: Any = None) -> Any:
        value = getattr(method, attr, None)
        if value is not None:
            return value
        func = getattr(method, "__func__", None)
        return getattr(func, attr, default)

    def _materialize_endpoint_info(
        self, attr_name: str, method, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        tactic_meta = self._callable_attr(method, "_lllm_tactic_endpoint")
        if not tactic_meta:
            return info

        from lllm.core.tactic_tool import build_tactic_endpoint_info

        return build_tactic_endpoint_info(
            tactic_meta["tactic_url"],
            runtime=getattr(self, "_runtime", None),
            base_namespace=getattr(self, "_resource_namespace", None),
            endpoint=tactic_meta.get("endpoint") or attr_name,
            category=tactic_meta.get("category") or "tactics",
            name=tactic_meta.get("name"),
            description=tactic_meta.get("description"),
            response=tactic_meta.get("response"),
            method=tactic_meta.get("method") or "TACTIC",
        )

    def _endpoint_methods(self):
        """
        Yield ``(attr_name, method, endpoint_info)`` triples for every method
        decorated with :func:`BaseProxy.endpoint`.
        """
        for name, method in inspect.getmembers(self, predicate=callable):
            info = self._callable_attr(method, "endpoint_info")
            if info:
                info = self._materialize_endpoint_info(name, method, dict(info))
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
        directory.sort(
            key=lambda item: ((item.get("category") or ""), item.get("endpoint") or "")
        )
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


class ProxyManager:
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
        runtime: Runtime = None,
    ):
        self._runtime = runtime or get_default_runtime()

        self.activate_proxies = activate_proxies or []
        self.cutoff_date = cutoff_date
        self.deploy_mode = deploy_mode
        self.proxies = {}
        self._load_registered_proxies()

    def _load_registered_proxies(self):
        for qk, node in self._runtime.iter_items("proxy"):
            proxy_cls = node.value
            if self.activate_proxies:
                # Match against qualified key, bare key, or _proxy_path
                matched = any(
                    act == qk
                    or act == node.key
                    or act == getattr(proxy_cls, "_proxy_path", None)
                    for act in self.activate_proxies
                )
                if not matched:
                    continue
            try:
                instance = proxy_cls(
                    cutoff_date=self.cutoff_date,
                    activate_proxies=self.activate_proxies,
                    deploy_mode=self.deploy_mode,
                    runtime=self._runtime,
                )
            except TypeError:
                instance = proxy_cls()
            instance._runtime = self._runtime
            instance._qualified_key = node.qualified_key
            instance._resource_namespace = node.namespace
            # Store under the bare key (what existing code expects for dispatch)
            self.proxies[node.key] = instance

    def register(self, name: str, proxy_cls: Any):
        """Register (or override) a proxy implementation at runtime."""
        if name in self.proxies:
            logger.warning(
                "Proxy '%s' already instantiated, overwriting instance", name
            )
        try:
            instance = proxy_cls(
                cutoff_date=self.cutoff_date,
                activate_proxies=self.activate_proxies,
                deploy_mode=self.deploy_mode,
                runtime=self._runtime,
            )
        except TypeError:
            instance = proxy_cls(
                self.activate_proxies, self.cutoff_date, self.deploy_mode
            )
        instance._runtime = self._runtime
        instance._qualified_key = getattr(proxy_cls, "_qualified_key", None)
        instance._resource_namespace = getattr(proxy_cls, "_resource_namespace", None)
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
                        type_name = (
                            getattr(type_hint, "__name__", str(type_hint))
                            if type_hint
                            else "Any"
                        )
                        if isinstance(example, (list, dict)):
                            example_preview = str(example)[:60]
                        else:
                            example_preview = example
                        lines.append(
                            f"    - `{param_name}` ({type_name}) e.g. {example_preview}"
                        )
            sections.append("\n".join(lines).strip())
        return "\n\n".join(sections).strip()

    def _resolve(self, endpoint: str) -> tuple[str, str]:
        if "." in endpoint:
            parts = endpoint.split(".", 1)
            return parts[0], parts[1]
        path_parts = endpoint.split("/")
        if len(path_parts) < 2:
            raise ValueError(
                f"Invalid endpoint '{endpoint}'. Use '<proxy>.<method>' or '<proxy>/<method>'."
            )
        return "/".join(path_parts[:-1]), path_parts[-1]

    def __call__(self, endpoint: str, *args, **kwargs):
        """Dispatch ``proxy_path.endpoint_name`` or ``proxy_path/endpoint`` to the proxy."""
        proxy_name, func_name = self._resolve(endpoint)
        if proxy_name not in self.proxies:
            raise KeyError(
                f"Proxy '{proxy_name}' not registered. Available: {list(self.proxies.keys())}"
            )
        proxy = self.proxies[proxy_name]
        if not hasattr(proxy, func_name):
            raise AttributeError(f"Proxy '{proxy_name}' has no endpoint '{func_name}'")
        handler = getattr(proxy, func_name)
        return handler(*args, **kwargs)


def ProxyRegistrator(path: str, name: str, description: str, runtime: Runtime = None):
    runtime = runtime or get_default_runtime()

    def decorator(cls):
        cls._proxy_path = path
        cls._proxy_name = name
        cls._proxy_description = description
        runtime.register_proxy(path, cls, overwrite=True)
        return cls

    return decorator


def register_proxy(name: str, proxy_cls, overwrite: bool = False):
    get_default_runtime().register_proxy(name, proxy_cls, overwrite)
