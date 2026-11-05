"""Regression tests for the second shell fix pass (review findings 7, 26, 33, 34)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lllm.network import LocalStore
from lllm.shell import Strategy, create_strategy_app
from lllm.shell.cli import main


def _error(response) -> dict[str, Any]:
    return response.json()["detail"]["error"]


# Finding 7: the channel routes served any resource, so a GET ran a tactic.


@pytest.mark.parametrize("kind", ["tactic", "service"])
def test_channel_routes_serve_only_channel_resources(kind: str):
    calls = []

    def deploy(input_value, context=None):
        calls.append(input_value)
        return {"deployed": True}

    strategy = Strategy("srv").resource(kind, "deploy", handler=deploy)
    client = TestClient(create_strategy_app(strategy))

    query = client.get("/channels/deploy/events")
    append = client.post("/channels/deploy/events", json={"input": {"payload": 1}})

    assert query.status_code == append.status_code == 404
    for response in (query, append):
        assert _error(response)["type"] == "ResourceNotFoundError"
        assert _error(response)["message"] == "Strategy channel not found: deploy"
        assert _error(response)["endpoint"] == "channel"
        assert _error(response)["metadata"] == {"resource": "deploy"}
    assert calls == []
    # The resource is still served by its own route.
    assert client.post("/resources/deploy/invoke", json={"input": 1}).status_code == 200
    assert calls == [1]


def test_channel_routes_still_call_a_channel_handler():
    seen = []

    def feed(input_value=None, *, context=None):
        seen.append(input_value)
        return {"op": input_value["op"]}

    client = TestClient(create_strategy_app(Strategy("srv").channel("feed", handler=feed)))

    assert client.get("/channels/feed/events").json()["output"] == {"op": "query"}
    appended = client.post("/channels/feed/events", json={"input": {"payload": 1}})
    assert appended.json()["output"] == {"op": "append"}
    assert seen[1] == {"payload": 1, "op": "append"}


# Finding 33: a non-object input to POST /channels/{name}/events was replaced
# with {}, appending an event without the data that was sent.


@pytest.mark.parametrize(
    "body",
    [
        {"input": "hello world"},
        {"input": ["x", "y"]},
        {"input": 3},
        {"input": None},
        {"payload": "outside the input envelope"},
    ],
)
def test_channel_append_rejects_a_non_object_input(tmp_path: Path, body: dict[str, Any]):
    store = LocalStore(tmp_path / "store")
    strategy = Strategy("srv").channel("log").attach_store(store)
    client = TestClient(create_strategy_app(strategy))

    response = client.post("/channels/log/events", json=body)

    assert response.status_code == 400
    assert _error(response) == {
        "type": "SchemaError",
        "message": "input must be an object with the event fields",
        "tactic": None,
        "endpoint": "channel",
        "request_id": None,
        "metadata": {"resource": "log"},
    }
    assert store.query_events("log") == ()
    appended = client.post("/channels/log/events", json={"input": {"payload": "kept"}})
    assert appended.status_code == 200
    assert [event.payload for event in store.query_events("log")] == ["kept"]


# Finding 26: keyword-only handlers were passed the input positionally.


def test_keyword_only_handlers_are_called_without_the_input():
    async def kwonly(*, context=None, verbose=False):
        return {"context": context, "verbose": verbose}

    strategy = Strategy("srv").service("kwonly", handler=kwonly)
    strategy.service("plain", handler=lambda *, verbose=False: {"verbose": verbose})
    strategy.runner(lambda *, context=None, dry_run=False: {"ran": True, "dry_run": dry_run})
    client = TestClient(create_strategy_app(strategy))

    assert asyncio.run(strategy.invoke_resource("kwonly", {"x": 1}, context={"a": 1})) == {
        "context": {"a": 1},
        "verbose": False,
    }
    assert asyncio.run(strategy.arun({"x": 1})) == {"ran": True, "dry_run": False}
    invoked = client.post("/resources/kwonly/invoke", json={"input": None})
    assert invoked.status_code == 200
    assert invoked.json()["output"] == {"context": {}, "verbose": False}
    plain = client.post("/resources/plain/invoke", json={"input": 1})
    assert plain.json()["output"] == {"verbose": False}
    assert client.post("/run", json={"input": 1}).json()["output"] == {
        "ran": True,
        "dry_run": False,
    }


def test_keyword_only_tactic_handlers_run_on_both_tactic_routes():
    async def kwonly(*, context=None, verbose=False):
        return {"context": type(context).__name__, "verbose": verbose}

    strategy = Strategy("srv").tactic("kwonly", handler=kwonly)
    strategy.tactic("takes_input", handler=lambda value, *, verbose=False: {"value": value})
    client = TestClient(create_strategy_app(strategy))

    for route in ("/tactics/kwonly/run", "/resources/kwonly/invoke"):
        response = client.post(route, json={"input": 7})
        assert response.status_code == 200
        assert response.json()["output"] == {"context": "CallContext", "verbose": False}
    # A positional parameter still takes the input, keyword-only extras or not.
    for route in ("/tactics/takes_input/run", "/resources/takes_input/invoke"):
        assert client.post(route, json={"input": 7}).json()["output"] == {"value": 7}


# Finding 34: a name StrategyInfo rejects was accepted, then GET /strategy was a 500.


@pytest.mark.parametrize("name", ["My Strategy", "a/b", "a:b", "x%y", "", "..", None])
def test_strategy_rejects_a_name_its_info_cannot_carry(name):
    with pytest.raises(ValueError, match="strategy.name must be a non-empty path segment"):
        Strategy(name)


def test_strategy_info_is_served_for_a_valid_name():
    strategy = Strategy("my-strategy.v2")
    client = TestClient(create_strategy_app(strategy))

    assert strategy.info().name == "my-strategy.v2"
    assert client.get("/strategy").json()["name"] == "my-strategy.v2"


def test_cli_reports_an_invalid_strategy_name(tmp_path: Path, capsys):
    (tmp_path / "strategy.py").write_text(
        "from lllm.shell import Strategy\nstrategy = Strategy('My Strategy')\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        main(["inspect", str(tmp_path)])

    assert exit_info.value.code == 2
    assert "strategy.name must be a non-empty path segment" in capsys.readouterr().err
