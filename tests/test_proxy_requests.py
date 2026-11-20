"""Proxy requests: the HTTP cache honors use_cache and never stores POST
responses, unknown proxy names are reported, POST endpoints send query-param
keys in the query, and tactic endpoints accept any input."""

import json
import logging
from urllib.parse import parse_qs, urlsplit

import pytest

from lllm.runtimes.native import Runtime
from lllm.runtimes.native import utils as U
from lllm.runtimes.native.proxies import BaseProxy, ProxyManager


class FakeResponse:
    status_code = 200

    def __init__(self, data):
        self.data = data
        self.text = json.dumps(data)

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


@pytest.fixture
def http(monkeypatch, tmp_path):
    """Record requests instead of sending them, with an empty cache folder."""

    import requests

    sent = []

    def get(url, params=None, headers=None):
        sent.append(("GET", url, dict(params or {})))
        return FakeResponse({"request": len(sent)})

    def post(url, json=None, headers=None):
        sent.append(("POST", url, dict(json or {})))
        return FakeResponse({"request": len(sent)})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(U, "CACHE_DIR", str(tmp_path))
    return sent


def cache_files(tmp_path):
    return sorted(path.name for path in tmp_path.rglob("*.json"))


URL = "https://api.example.test/v1/rates"


def test_get_responses_are_cached_by_default(http, tmp_path):
    assert U.call_api(URL, {"base": "USD"}) == {"request": 1}
    assert U.call_api(URL, {"base": "USD"}) == {"request": 1}

    assert len(http) == 1
    assert len(cache_files(tmp_path)) == 1


def test_use_cache_false_neither_reads_nor_writes_the_cache(http, tmp_path):
    U.call_api(URL, {"base": "USD"})  # cached

    assert U.call_api(URL, {"base": "USD"}, use_cache=False) == {"request": 2}
    assert U.call_api(URL, {"base": "EUR"}, use_cache=False) == {"request": 3}
    assert U.call_api(URL, {"base": "USD"}) == {"request": 1}  # not overwritten

    assert len(cache_files(tmp_path)) == 1


def test_post_responses_are_never_cached(http, tmp_path):
    assert U.call_api_post(URL, {"text": "hi"}) == {"request": 1}
    assert U.call_api_post(URL, {"text": "hi"}) == {"request": 2}

    assert cache_files(tmp_path) == []


def test_plain_text_responses_are_returned_and_not_cached(http, tmp_path):
    response = U.call_api(URL, {"input": "pi"}, json_response=False)

    assert response.text == '{"request": 1}'
    assert cache_files(tmp_path) == []


class RatesProxy(BaseProxy):
    base_url = "https://api.example.test/v1"
    api_key_name = "apikey"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = "secret"

    @BaseProxy.endpoint(
        category="rates",
        endpoint="rates",
        description="Latest rates.",
        params={"base": (str, "USD")},
        response={},
    )
    def rates(self, params):
        return params

    @BaseProxy.endpoint(
        category="rates",
        endpoint="convert",
        description="Convert an amount.",
        params={"amount": (float, 1.0)},
        response={},
        method="POST",
    )
    def convert(self, params):
        return params


class UncachedRatesProxy(RatesProxy):
    def __init__(self, **kwargs):
        super().__init__(use_cache=False, **kwargs)


def test_a_proxy_with_use_cache_false_sends_every_request(http, tmp_path):
    proxy = UncachedRatesProxy(runtime=Runtime())

    assert proxy.call("rates", {"base": "USD"}) == {"request": 1}
    assert proxy.call("rates", {"base": "USD"}) == {"request": 2}
    assert cache_files(tmp_path) == []
