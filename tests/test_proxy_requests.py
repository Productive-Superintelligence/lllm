"""Proxy requests: the HTTP cache honors use_cache and never stores POST
responses, unknown proxy names are reported, POST endpoints send query-param
keys in the query, and tactic endpoints accept any input."""

import importlib
import importlib.util
import json
import logging
import sqlite3
import sys
import threading
import unittest.mock
from array import array
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch, seal
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import BaseModel
from pydantic import v1 as pydantic_v1

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


@pytest.fixture
def exa(monkeypatch, tmp_path):
    """The built-in Exa proxy module, with a fake exa_py client. Its
    registration goes to a throwaway default runtime."""

    from lllm.runtimes.native.core import runtime as runtime_module

    sent = []

    @dataclass
    class Result:
        url: str

    class Exa:
        def __init__(self, api_key):
            pass

        def get_contents(self, urls, text):
            sent.append(urls)
            return SimpleNamespace(
                results=[Result(url) for url in urls],
                autoprompt_string=None,
                resolved_search_type=None,
                auto_date=None,
                cost_dollars=None,
            )

    name = "lllm.runtimes.native.proxies.builtin.exa_proxy"
    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=Exa))
    monkeypatch.setattr(runtime_module, "_default_runtime", Runtime())
    monkeypatch.setattr(U, "CACHE_DIR", str(tmp_path))
    sys.modules.pop(name, None)
    yield importlib.import_module(name), sent
    sys.modules.pop(name, None)


def test_the_exa_proxy_honors_use_cache(exa, tmp_path, monkeypatch):
    module, sent = exa
    info = {"endpoint": "contents"}
    cached = module.ExaProxy(runtime=Runtime())
    uncached = module.ExaProxy(use_cache=False, runtime=Runtime())

    first = cached._call_api("contents", {"urls": ["https://a.test"]}, info, {})
    assert cached._call_api("contents", {"urls": ["https://a.test"]}, info, {}) == first
    assert len(sent) == 1
    assert len(cache_files(tmp_path)) == 1

    reads = []
    load = U.load_cache_by_key
    monkeypatch.setattr(U, "load_cache_by_key", lambda *a: reads.append(a) or load(*a))
    uncached._call_api("contents", {"urls": ["https://a.test"]}, info, {})
    uncached._call_api("contents", {"urls": ["https://b.test"]}, info, {})
    assert reads == []  # never read
    assert len(sent) == 3
    assert len(cache_files(tmp_path)) == 1  # never written


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

    @BaseProxy.endpoint(
        category="rates",
        endpoint="history",
        description="Rates between two dates.",
        params={"start": (str, "2024-01-01"), "end": (str, "2024-12-31")},
        response={},
    )
    def history(self, params):
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


def test_keyword_params_join_the_positional_ones():
    proxies = tactic_tools()

    assert proxies("tools/repeat", "ab", times=3) == "ababab"
    assert proxies("tools/repeat", {"text": "ab"}, times=3) == "ababab"
    assert proxies("tools/repeat", params={"text": "ab"}, times=3) == "ababab"
    assert proxies("tools/repeat", text="ab", times=3) == "ababab"


class PlainProxy:
    """A proxy that does not subclass BaseProxy."""

    def __init__(self, **kwargs):
        pass

    def call(self, path, params=None):
        return path, params


class OwnCallProxy(BaseProxy):
    """A BaseProxy that routes calls itself."""

    def call(self, path, params=None):
        return path, params


@pytest.mark.parametrize(
    ("proxy_cls", "lone_value_error"),
    [(PlainProxy, (TypeError, "as a dict")), (OwnCallProxy, (TypeError, "as a dict"))],
)
def test_keyword_params_reach_proxies_with_their_own_call(proxy_cls, lone_value_error):
    proxies = ProxyManager(runtime=Runtime())
    proxies.register("own", proxy_cls)

    assert proxies("own/anything", {"a": 1}, b=2) == ("anything", {"a": 1, "b": 2})
    assert proxies("own/anything", params={"a": 1}, b=2) == ("anything", {"a": 1, "b": 2})
    assert proxies("own/anything", b=2) == ("anything", {"b": 2})
    assert proxies("own/anything", [("a", 1)], b=2) == ("anything", {"a": 1, "b": 2})
    error, match = lone_value_error  # a lone value needs an endpoint to name it after
    with pytest.raises(error, match=match):
        proxies("own/anything", "lone", b=2)


def test_an_unknown_endpoint_with_a_lone_value_and_keywords_says_so():
    with pytest.raises(AttributeError, match="No endpoint 'missing'"):
        tactic_tools()("tools/missing", "lone", times=2)


class Delegating:
    """A proxy that hands every attribute to another one."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


class InstanceCall:
    """A proxy whose call is an attribute of the instance."""

    def __init__(self):
        self.call = lambda path, params=None: (path, params)


@pytest.mark.parametrize(
    "kind",
    ["delegating", "bound elsewhere", "spied", "mocked", "mocked with spec", "mock", "instance call"],
)
def test_a_lone_value_with_keywords_on_other_kinds_of_proxy(kind):
    proxies = tactic_tools()
    tools = proxies.proxies["tools"]

    if kind == "spied":  # mock.patch.object(..., wraps=...), mocker.spy
        with patch.object(tools, "call", wraps=tools.call):
            assert proxies("tools/repeat", "ab", times=3) == "ababab"
    elif kind.startswith("mocked"):  # the mock gets the documented params
        options = {"spec": True} if kind == "mocked with spec" else {"return_value": "mocked"}
        with patch.object(tools, "call", **options) as mocked:
            listed = [entry["callable"] for entry in tools.endpoint_directory()]
            proxies("tools/repeat", "ab", times=3)
        assert listed == ["join", "repeat", "shout"]  # the mock is not an endpoint
        mocked.assert_called_once_with("repeat", {"text": "ab", "times": 3})
    elif kind in ("delegating", "bound elsewhere"):  # BaseProxy.call names the value
        other = Delegating(tools) if kind == "delegating" else SimpleNamespace(call=tools.call)
        proxies.proxies["other"] = other
        assert proxies("other/repeat", "ab", times=3) == "ababab"
    else:
        proxies.proxies["other"] = MagicMock() if kind == "mock" else InstanceCall()
        with pytest.raises(TypeError, match=r"'other/anything': pass the params as a dict"):
            proxies("other/anything", "lone", b=2)


def test_a_spied_tactic_endpoint_is_still_found_and_called():
    proxies = tactic_tools()
    tools = proxies.proxies["tools"]

    with patch.object(tools, "repeat", wraps=tools.repeat) as spy:
        assert proxies("tools/repeat", {"text": "ab", "times": 3}) == "ababab"
        # A lone value is named from the tactic's materialized params.
        assert proxies("tools/repeat", "ab", times=3) == "ababab"

    assert spy.call_count == 2


def test_a_spy_of_a_spy_of_a_tactic_endpoint_is_found_and_called():
    proxies = tactic_tools()
    tools = proxies.proxies["tools"]

    with patch.object(tools, "repeat", wraps=tools.repeat) as inner:  # a fixture's spy
        with patch.object(tools, "repeat", wraps=tools.repeat) as outer:  # the test's
            listed = [entry["callable"] for entry in tools.endpoint_directory()]
            assert proxies("tools/repeat", {"text": "ab", "times": 3}) == "ababab"

    assert listed == ["join", "repeat", "shout"]
    inner.assert_called_once()
    outer.assert_called_once()


MOCKED_INFO = {
    "category": "rates",
    "endpoint": "mocked",
    "description": "A mocked endpoint.",
    "params": {"base": (str, "USD")},
    "response": {},
    "method": "GET",
}


def test_a_wrapping_mock_with_its_own_endpoint_info_is_an_endpoint(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    proxy.mocked = MagicMock(wraps=lambda params: params, endpoint_info=MOCKED_INFO)

    assert "mocked" in [entry["callable"] for entry in proxy.endpoint_directory()]
    proxy.call("mocked", {"base": "AUD"})

    assert http == [
        ("GET", "https://api.example.test/v1/mocked", {"base": "AUD", "apikey": "secret"})
    ]


def test_a_spys_own_endpoint_info_wins_over_the_wrapped_one(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        spy.endpoint_info = {**RatesProxy.rates.endpoint_info, "endpoint": "rates/v2"}
        proxy.call("rates/v2", {"base": "USD"})

    assert http == [
        ("GET", "https://api.example.test/v1/rates/v2", {"base": "USD", "apikey": "secret"})
    ]


def rates_info(path):
    return {**RatesProxy.rates.endpoint_info, "endpoint": path}


def test_a_property_mock_on_a_spys_type_gives_its_endpoint_info(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        type(spy).endpoint_info = PropertyMock(return_value=rates_info("rates/v3"))
        proxy.call("rates/v3", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/v3"


class EndpointSpy(MagicMock):
    """A spy class that carries its own endpoint metadata."""

    endpoint_info = rates_info("rates/spied")


def test_a_spy_class_with_its_own_endpoint_info_routes_by_it(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    proxy.spied = EndpointSpy(wraps=proxy.rates)

    proxy.call("rates/spied", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/spied"


def test_deleting_a_spys_endpoint_info_hides_the_endpoint():
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        del spy.endpoint_info
        listed = [entry["callable"] for entry in proxy.endpoint_directory()]

    assert listed == ["convert", "history"]


def other_mock_library():
    """Another mock library, as the standalone mock package is: a second
    copy of unittest.mock's code, with its own classes and sentinels."""

    spec = importlib.util.spec_from_file_location("lllm_test_other_mock", unittest.mock.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OTHER_MOCK = other_mock_library()


def standalone_mock_class():
    return pytest.importorskip("mock").MagicMock


OTHER_LIBRARIES = pytest.mark.parametrize(
    "make_class", [lambda: OTHER_MOCK.MagicMock, standalone_mock_class], ids=["simulated", "mock"]
)


@OTHER_LIBRARIES
def test_deleting_endpoint_info_on_another_mock_librarys_spy_hides_it(make_class):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", new=make_class()(wraps=proxy.rates)) as spy:
        del spy.endpoint_info
        listed = [entry["callable"] for entry in proxy.endpoint_directory()]

    assert listed == ["convert", "history"]


@OTHER_LIBRARIES
def test_another_mock_librarys_spy_is_read_through_and_routed(http, make_class):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", new=make_class()(wraps=proxy.rates)) as spy:
        proxy.call("rates", {"base": "USD"})

    spy.assert_called_once()
    assert http == [
        ("GET", "https://api.example.test/v1/rates", {"base": "USD", "apikey": "secret"})
    ]


def test_a_property_mock_on_a_spys_type_wins_over_a_deletion(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        del spy.endpoint_info
        type(spy).endpoint_info = PropertyMock(return_value=rates_info("rates/v4"))
        proxy.call("rates/v4", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/v4"


def test_a_property_mock_on_a_spys_type_wins_over_its_own_value(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        spy.endpoint_info = rates_info("rates/own")
        type(spy).endpoint_info = PropertyMock(return_value=rates_info("rates/v5"))
        proxy.call("rates/v5", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/v5"


@pytest.mark.parametrize("hide", ["seal", "spec"])
def test_a_sealed_spy_or_one_whose_spec_lacks_endpoint_info_is_hidden(hide):
    proxy = UncachedRatesProxy(runtime=Runtime())
    options = {"spec": lambda params: params} if hide == "spec" else {}

    with patch.object(proxy, "rates", wraps=proxy.rates, **options) as spy:
        if hide == "seal":
            seal(spy)
        listed = [entry["callable"] for entry in proxy.endpoint_directory()]

    assert listed == ["convert", "history"]  # as in 0.4.0


class AnsweringSpy(MagicMock):
    """A spy class whose own __getattr__ answers endpoint_info."""

    def __getattr__(self, name):
        if name == "endpoint_info":
            return rates_info("rates/answered")
        return super().__getattr__(name)


def test_a_spy_class_whose_getattr_answers_endpoint_info_is_honored(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    proxy.answered = AnsweringSpy(wraps=proxy.rates)

    proxy.call("rates/answered", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/answered"


def test_an_inner_spys_deletion_hides_the_endpoint_through_an_outer_spy():
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as inner:
        del inner.endpoint_info
        with patch.object(proxy, "rates", wraps=proxy.rates):
            listed = [entry["callable"] for entry in proxy.endpoint_directory()]

    assert listed == ["convert", "history"]


def test_an_inner_spys_property_mock_routes_through_an_outer_spy(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as inner:
        type(inner).endpoint_info = PropertyMock(return_value=rates_info("rates/inner"))
        with patch.object(proxy, "rates", wraps=proxy.rates):
            proxy.call("rates/inner", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/inner"


def test_a_wrapping_mock_with_its_own_tactic_metadata_runs_as_a_tactic(http):
    runtime = Runtime()
    runtime.register_tactic("shout", Shout)
    proxy = MixedProxy(runtime=runtime)  # has a base_url and an API key
    proxy.fake = MagicMock(
        wraps=lambda params: params["text"].upper(),
        endpoint_info=dict(MixedProxy.shout.endpoint_info),
        _lllm_tactic_endpoint=dict(MixedProxy.shout._lllm_tactic_endpoint),
    )

    assert "fake" in [entry["callable"] for entry in proxy.endpoint_directory()]
    assert proxy.call("fake", {"text": "hi"}) == "HI"
    assert proxy.call("fake", "hey") == "HEY"  # named from the tactic's params
    assert http == []  # no request, so no API key sent anywhere


def test_an_inner_spys_own_endpoint_info_wins_through_an_outer_spy(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as fixture_spy:
        fixture_spy.endpoint_info = rates_info("rates/v2")
        with patch.object(proxy, "rates", wraps=proxy.rates):  # the test's own spy
            proxy.call("rates/v2", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates/v2"


class Forwarding:
    """A wrapper whose __getattr__ forwards everything, _mock_wraps too."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)


def test_a_forwarding_wrapper_is_read_as_is_not_followed(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    proxy.mocked = Forwarding(MagicMock(wraps=lambda params: params, endpoint_info=MOCKED_INFO))

    assert "mocked" in [entry["callable"] for entry in proxy.endpoint_directory()]
    proxy.call("mocked", {"base": "AUD"})

    assert http[0][1] == "https://api.example.test/v1/mocked"


def test_a_forwarding_wrapper_over_a_spy_gives_the_metadata_the_spy_wraps(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    spy = MagicMock(wraps=proxy.rates)
    proxy.forwarded = Forwarding(spy)  # 0.4.0 listed it, through the spy's child mock

    assert "forwarded" in [entry["callable"] for entry in proxy.endpoint_directory()]
    proxy.call("forwarded", {"base": "USD"})  # endpoint "rates", the spy's

    spy.assert_called_once()
    assert http[0][1] == "https://api.example.test/v1/rates"


@pytest.mark.parametrize("depth", [2, 3])
def test_a_forwarding_wrapper_over_a_spy_of_a_spy_gives_the_metadata_they_wrap(http, depth):
    proxy = UncachedRatesProxy(runtime=Runtime())
    spied = proxy.rates
    for _ in range(depth):  # a spy of a spy (of a spy)
        spied = MagicMock(wraps=spied)
    proxy.forwarded = Forwarding(spied)

    assert "forwarded" in [entry["callable"] for entry in proxy.endpoint_directory()]
    proxy.call("forwarded", {"base": "USD"})

    assert http[0][1] == "https://api.example.test/v1/rates"


def test_a_tactic_endpoint_behind_two_spies_and_a_forwarding_wrapper_runs():
    proxies = tactic_tools()
    tools = proxies.proxies["tools"]

    with patch.object(tools, "repeat", wraps=tools.repeat) as fixture_spy:
        with patch.object(tools, "repeat", wraps=tools.repeat) as test_spy:
            tools.forwarded = Forwarding(test_spy)
            assert proxies("tools/forwarded", {"text": "ab", "times": 3}) == "ababab"

    fixture_spy.assert_called_once()
    test_spy.assert_called_once()


def test_a_cycle_of_wrapping_mocks_is_not_an_endpoint():
    proxy = UncachedRatesProxy(runtime=Runtime())
    first, second = MagicMock(), MagicMock()
    first.__dict__["_mock_wraps"], second.__dict__["_mock_wraps"] = second, first
    proxy.cyclic = first  # reading one of these mocks would recurse in Mock

    assert [entry["callable"] for entry in proxy.endpoint_directory()] == [
        "convert", "history", "rates",
    ]


def test_a_child_that_wraps_itself_behind_a_forwarding_wrapper_is_not_an_endpoint():
    proxy = UncachedRatesProxy(runtime=Runtime())
    spy = MagicMock(wraps=proxy.rates)
    child = spy.endpoint_info
    child._mock_wraps = child  # a plain assignment: it lands in __dict__
    proxy.looped = Forwarding(spy)
    listed = []

    # Unwrapping the child would never end without its bound: run it on a
    # thread, so that this test fails instead of hanging.
    worker = threading.Thread(
        target=lambda: listed.extend(entry["callable"] for entry in proxy.endpoint_directory()),
        daemon=True,
    )
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "reading the looped child did not finish"
    assert listed == ["convert", "history", "rates"]


class MixedProxy(UncachedRatesProxy):
    """An HTTP proxy with a tactic endpoint too."""

    shout = BaseProxy.tactic_endpoint("shout", config={})


def test_a_spied_tactic_endpoint_on_an_http_proxy_still_runs_the_tactic(http):
    runtime = Runtime()
    runtime.register_tactic("shout", Shout)
    proxy = MixedProxy(runtime=runtime)

    with patch.object(proxy, "shout", wraps=proxy.shout) as spy:
        assert proxy.call("shout", {"text": "hi"}) == "HI"

    spy.assert_called_once()
    assert http == []  # run as a tactic: no request


def test_a_spied_http_endpoint_still_sends_its_request(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with patch.object(proxy, "rates", wraps=proxy.rates) as spy:
        proxy.call("rates", {"base": "USD"})

    spy.assert_called_once()
    assert http == [
        ("GET", "https://api.example.test/v1/rates", {"base": "USD", "apikey": "secret"})
    ]


def test_a_mock_with_real_endpoint_info_is_an_http_endpoint(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    proxy.mocked = MagicMock(  # its _lllm_tactic_endpoint is made up, not metadata
        return_value={"base": "AUD"},
        endpoint_info={
            "category": "rates",
            "endpoint": "mocked",
            "description": "A mocked endpoint.",
            "params": {"base": (str, "USD")},
            "response": {},
            "method": "GET",
        },
    )

    proxy.call("mocked", {"base": "AUD"})

    assert http == [
        ("GET", "https://api.example.test/v1/mocked", {"base": "AUD", "apikey": "secret"})
    ]


def test_more_than_one_positional_param_is_an_error():
    proxies = tactic_tools()

    with pytest.raises(TypeError, match="one positional params value"):
        proxies("tools/repeat", "ab", 3)


def test_an_http_endpoint_with_one_param_takes_a_lone_value_too(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    proxy.call("rates", "EUR")

    assert http[0][2] == {"base": "EUR", "apikey": "secret"}


class Base(BaseModel):
    base: str


class Text(BaseModel):
    text: str


class V1Base(pydantic_v1.BaseModel):
    base: str


def sent_params(http):
    return [{k: v for k, v in sent.items() if k != "apikey"} for _m, _u, sent in http]


def test_models_and_any_iterable_of_pairs_are_params(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    for params in (
        Base(base="JPY"),
        V1Base(base="NZD"),  # iterates as pairs
        [("base", "GBP")],
        (["base", "CHF"],),
        {"base": "SEK"}.items(),
        zip(["base"], ["NOK"]),
        (pair for pair in [("base", "DKK")]),
        {("base", "AUD")},
        deque([("base", "CAD")]),
    ):
        proxy.call("rates", params)

    assert [sent["base"] for sent in sent_params(http)] == [
        "JPY", "NZD", "GBP", "CHF", "SEK", "NOK", "DKK", "AUD", "CAD",
    ]
    assert tactic_tools()("tools/shout", Text(text="hi")) == "HI"


def test_what_dict_takes_is_params_as_in_0_4_0(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row

    proxy.call("rates", connection.execute("select 'JPY' as base").fetchone())  # keys()
    proxy.call("rates", ["ab", "cd"])  # two-letter strings are pairs to dict()

    assert sent_params(http) == [{"base": "JPY"}, {"a": "b", "c": "d"}]


def test_anything_else_reaches_the_endpoint_as_it_is(http):
    proxy = UncachedRatesProxy(runtime=Runtime())
    values = ({"USD", "EUR"}, range(3), array("i", [1, 2]), ("USD", "EUR"))

    for value in values:
        proxy.call("rates", value)
    proxy.call("rates", iter(["USD", "EUR"]))  # a one-shot iterator: read once

    sent = [params["base"] for params in sent_params(http)]
    assert all(got is value for got, value in zip(sent, values))
    assert sent[-1] == ["USD", "EUR"]


def test_a_lone_value_error_names_the_type_the_caller_passed(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with pytest.raises(TypeError, match=r"\(start, end\), not list_iterator\."):
        proxy.call("history", iter(["2024"]))


class Ambiguous:
    """Like a numpy array of two items: iterable, with no truth value."""

    def __iter__(self):
        return iter([1, 2])

    def __bool__(self):
        raise ValueError("the truth value is ambiguous")


def test_a_lone_value_without_a_truth_value_gets_the_usual_error(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    with pytest.raises(TypeError, match=r"\(start, end\), not Ambiguous\."):
        proxy.call("history", Ambiguous())


def test_none_and_empty_containers_mean_no_params(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    for params in (None, {}, [], (), set(), iter([])):
        proxy.call("rates", params)

    assert sent_params(http) == [{}] * 6


def test_falsy_values_are_a_lone_value_only_where_the_endpoint_takes_one(http):
    proxy = UncachedRatesProxy(runtime=Runtime())

    for params in (0, 0.0, False, ""):
        proxy.call("rates", params)  # one parameter: the lone value
        proxy.call("history", params)  # two optional ones: no params, as in 0.4.0

    assert sent_params(http) == [
        {"base": 0}, {}, {"base": 0.0}, {}, {"base": False}, {}, {"base": ""}, {},
    ]
    with pytest.raises(TypeError, match=r"\(start, end\), not str"):
        proxy.call("history", "2024")


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
