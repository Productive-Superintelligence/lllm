"""Proxy requests: the HTTP cache honors use_cache and never stores POST
responses, unknown proxy names are reported, POST endpoints send query-param
keys in the query, and tactic endpoints accept any input."""

import json
import logging
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import BaseModel

from lllm import Tactic
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


def test_names_in_activate_proxies_that_nothing_registered_are_reported(caplog):
    runtime = Runtime()
    runtime.register_proxy("rates", RatesProxy)

    with caplog.at_level(logging.WARNING, logger="lllm"):
        proxies = ProxyManager(activate_proxies=["rates", "weather"], runtime=runtime)

    assert proxies.available() == ["rates"]
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "'weather'" in record.getMessage()
    assert "rates" in record.getMessage()  # the proxies that are registered


def test_post_endpoints_send_a_query_param_key_in_the_query(http):
    proxy = RatesProxy(runtime=Runtime())

    proxy.call("convert", {"amount": 2.5})

    ((method, url, body),) = http
    assert method == "POST"
    assert url.startswith("https://api.example.test/v1/convert?")
    assert parse_qs(urlsplit(url).query) == {"apikey": ["secret"]}
    assert body == {"amount": 2.5}


def test_get_endpoints_send_a_query_param_key_with_the_params(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    proxy.call("rates", {"base": "USD"})

    assert http == [
        ("GET", "https://api.example.test/v1/rates", {"base": "USD", "apikey": "secret"})
    ]


class Shout(Tactic):
    name = "shout"

    def call(self, text: str) -> str:
        return text.upper()


class Repeat(Tactic):
    name = "repeat"

    def call(self, text: str, times: int = 2) -> str:
        return text * times


class Join(Tactic):
    name = "join"

    def call(self, first: str, second: str) -> str:
        return first + second


class TacticTools(BaseProxy):
    shout = BaseProxy.tactic_endpoint("shout", config={})
    repeat = BaseProxy.tactic_endpoint("repeat", config={})
    join = BaseProxy.tactic_endpoint("join", config={})


def tactic_tools():
    runtime = Runtime()
    for tactic in (Shout, Repeat, Join):
        runtime.register_tactic(tactic.name, tactic)
    proxies = ProxyManager(runtime=runtime)
    proxies.register("tools", TacticTools)
    return proxies


def test_a_tactic_endpoint_takes_a_lone_value_as_its_input():
    proxies = tactic_tools()

    assert proxies("tools/shout", "hi") == "HI"
    assert proxies("tools/shout", {"text": "hi"}) == "HI"
    assert proxies("tools/shout", text="hi") == "HI"
    assert proxies("tools/repeat", "ab") == "abab"  # the only required one


def test_a_lone_value_for_several_required_params_is_a_clear_error():
    proxies = tactic_tools()

    assert proxies("tools/join", {"first": "a", "second": "b"}) == "ab"
    with pytest.raises(TypeError, match=r"takes its params as a dict \(first, second\), not str"):
        proxies("tools/join", "ab")


def test_an_http_endpoint_with_one_param_takes_a_lone_value_too(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    proxy.call("rates", "EUR")

    assert http[0][2] == {"base": "EUR", "apikey": "secret"}


class Base(BaseModel):
    base: str


class Text(BaseModel):
    text: str


def test_models_pairs_and_empty_values_are_params_as_before(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    for params in (Base(base="JPY"), [("base", "GBP")], (["base", "CHF"],), "", [], None):
        proxy.call("rates", params)

    assert [sent for _method, _url, sent in http] == [
        {"base": "JPY", "apikey": "secret"},
        {"base": "GBP", "apikey": "secret"},
        {"base": "CHF", "apikey": "secret"},
        {"apikey": "secret"},
        {"apikey": "secret"},
        {"apikey": "secret"},
    ]
    assert tactic_tools()("tools/shout", Text(text="hi")) == "HI"


class MarkedProxy(BaseProxy):
    """Marks params like the built-in msd proxy: "$" for a path value, "#"
    for an item of a POST body."""

    base_url = "https://api.example.test/v1"

    def _call_api(self, url, params, endpoint_info, headers):
        return url, params

    @BaseProxy.endpoint(
        category="companies",
        endpoint="companies/{ticker}/kpi",
        description="A company's KPIs.",
        params={"$ticker*": (str, "AAPL")},
        response=[],
    )
    def kpi(self, params):
        return params

    @BaseProxy.endpoint(
        category="companies",
        endpoint="companies",
        description="Several companies.",
        params={"#tickers*": (list, ["AAPL"]), "startDate": (str, "2023-01-01")},
        response={},
        method="POST",
    )
    def companies(self, params):
        return params

    @BaseProxy.endpoint(
        category="companies",
        endpoint="compare/{left}",
        description="Compare two companies.",
        params={"$left*": (str, "AAPL"), "#right*": (str, "MSFT")},
        response={},
        method="POST",
    )
    def compare(self, params):
        return params


def test_lone_values_name_params_marked_with_dollar_or_hash():
    proxy = MarkedProxy(runtime=Runtime())

    assert proxy.call("companies/{ticker}/kpi", "MSFT") == (
        "https://api.example.test/v1/companies/MSFT/kpi",
        {},
    )
    assert proxy.call("companies", ["AAPL", "MSFT"]) == (
        "https://api.example.test/v1/companies",
        {"tickers": ["AAPL", "MSFT"]},
    )
    with pytest.raises(TypeError, match=r"takes its params as a dict \(left, right\), not str"):
        proxy.call("compare/{left}", "AAPL")
