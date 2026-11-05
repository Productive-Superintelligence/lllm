"""Proxies: documented endpoint paths reach the API, and only endpoints do."""

import pytest

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import BaseProxy, ProxyManager


class RecordingProxy(BaseProxy):
    """An HTTP-style proxy whose transport is recorded instead of sent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.example.test/v1"
        self.api_key_name = "apikey"
        self.api_key = "secret"
        self.sent = []

    def _call_api(self, url, params, endpoint_info, headers):
        self.sent.append((url, dict(params), dict(headers)))
        return {"url": url}

    @BaseProxy.endpoint(
        category="Search",
        endpoint="search-symbol",
        description="Find symbols.",
        params={"query": (str, "AAPL")},
        response=[],
    )
    def search_symbol(self, params):
        params.setdefault("limit", 5)
        return params

    @BaseProxy.endpoint(
        category="Companies",
        endpoint="companies/{ticker}/kpi",
        description="Company KPIs.",
        params={"ticker": (str, "AAPL")},
        response=[],
    )
    def company_kpi(self, params):
        return params


class HeaderKeyProxy(RecordingProxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key_name = "*x-api-key"


class LocalProxy(BaseProxy):
    """A proxy that computes results itself (no base_url)."""

    @BaseProxy.endpoint(
        category="Math",
        endpoint="math/double",
        description="Double a number.",
        params={"x": (int, 2)},
        response=[],
    )
    def double(self, params):
        return {"value": params["x"] * 2}


def manager(**proxies):
    proxy_manager = ProxyManager(runtime=Runtime())
    for name, proxy_cls in proxies.items():
        proxy_manager.register(name.replace("__", "/"), proxy_cls)
    return proxy_manager


def test_documented_paths_reach_the_api_with_the_key():
    proxies = manager(demo=RecordingProxy)

    proxies("demo/search-symbol", {"query": "LLLM"})

    assert proxies.proxies["demo"].sent == [
        (
            "https://api.example.test/v1/search-symbol",
            {"query": "LLLM", "limit": 5, "apikey": "secret"},
            {},
        )
    ]


def test_path_templates_fill_from_the_path_or_the_params():
    proxies = manager(demo=RecordingProxy)

    proxies("demo/companies/AAPL/kpi", {})
    proxies("demo/companies/{ticker}/kpi", {"ticker": "MSFT"})

    sent = proxies.proxies["demo"].sent
    assert [url for url, _params, _headers in sent] == [
        "https://api.example.test/v1/companies/AAPL/kpi",
        "https://api.example.test/v1/companies/MSFT/kpi",
    ]
    assert all("ticker" not in params for _url, params, _headers in sent)


def test_header_keys_and_method_name_fallback():
    proxies = manager(demo=HeaderKeyProxy)

    proxies("demo.search_symbol", {"query": "x"})

    url, params, headers = proxies.proxies["demo"].sent[0]
    assert headers == {"x-api-key": "secret"}
    assert "apikey" not in params and "x-api-key" not in params


def test_only_decorated_endpoints_are_reachable():
    proxies = manager(demo=RecordingProxy)

    for path in ("demo/auto_test", "demo/_call_api", "demo/missing"):
        with pytest.raises(AttributeError, match="No endpoint"):
            proxies(path, {})


def test_nested_proxy_names_resolve_by_longest_prefix():
    proxies = manager(data__demo=RecordingProxy, data=LocalProxy)

    proxies("data/demo/search-symbol", {"query": "x"})

    assert proxies.proxies["data/demo"].sent[0][0].endswith("/search-symbol")
    assert proxies("data/math/double", {"x": 21}) == {"value": 42}


def test_builtin_fmp_proxy_calls_its_documented_endpoint(monkeypatch):
    import lllm.runtimes.native.proxies.builtin.fmp_proxy as fmp_module

    calls = []
    monkeypatch.setenv("FMP_API_KEY", "fmp-key")
    monkeypatch.setattr(
        fmp_module.U,
        "call_api",
        lambda url, params, headers, use_cache=True: calls.append((url, params)) or [],
    )
    proxies = manager(fmp=fmp_module.FMPProxy)

    proxies("fmp/search-cusip", {"cusip": "037833100"})

    assert calls == [
        (
            "https://financialmodelingprep.com/stable/search-cusip",
            {"cusip": "037833100", "apikey": "fmp-key"},
        )
    ]
