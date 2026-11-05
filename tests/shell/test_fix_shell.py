"""Regression tests for shell fixes SH1-SH6."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lllm import Tactic
from lllm.network import (
    ChannelExistsError,
    ChannelNotFoundError,
    InvalidPayloadError,
    LocalStore,
    TacticProcessor,
)
from lllm.protocol import CallContext
from lllm.shell import Strategy, create_strategy_app, load_strategy
from lllm.shell.cli import main
from lllm.shell.strategy import ResourceNotFoundError, StrategyStore


def _package(root: Path, name: str = "analysis-pack", *, channels=("events",)) -> Path:
    module = root / "demo_pkg"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "tactics.py").write_text(
        """
def echo(input_value, *, context=None):
    return {
        "echo": input_value,
        "context_type": type(context).__name__,
        "metadata": getattr(context, "metadata", context),
        "tactic_ref": getattr(context, "tactic_ref", None),
    }
""".lstrip(),
        encoding="utf-8",
    )
    sections = "".join(
        f'\n[channels.{channel}]\nform = "log"\ndescription = "The {channel} stream."\n'
        for channel in channels
    )
    (root / "lllm.toml").write_text(
        f"""
[package]
org = "demo"
name = "{name}"
version = "0.1.0"
kind = "app"
description = "Demo package."

[tactics.echo]
entry = "demo_pkg.tactics:echo"
description = "Echo tactic."
{sections}""".lstrip(),
        encoding="utf-8",
    )
    return root


def _error(response) -> dict[str, Any]:
    return response.json()["detail"]["error"]


# SH1: channels declared with strategy.channel() are backed by the attached store.


def test_manual_channel_is_backed_by_the_attached_store(tmp_path: Path):
    store = LocalStore(tmp_path / "store")
    strategy = Strategy("manual")
    strategy.channel("log", description="Everything.")
    strategy.attach_store(store)
    client = TestClient(create_strategy_app(strategy))

    appended = client.post(
        "/channels/log/events",
        json={"input": {"payload": {"n": 1}, "kind": "note"}},
    )
    queried = client.get("/channels/log/events")
    info = client.post("/resources/log/invoke", json={"input": {"op": "info"}})

    assert appended.status_code == 200
    assert appended.json()["output"]["payload"] == {"n": 1}
    assert appended.json()["output"]["kind"] == "note"
    assert [event["payload"] for event in queried.json()["output"]] == [{"n": 1}]
    assert info.json()["output"]["description"] == "Everything."
    assert [event.payload for event in store.query_events("log")] == [{"n": 1}]


def test_store_backing_does_not_depend_on_declaration_order(tmp_path: Path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "log", "description": "Created by hand."})
    strategy = Strategy("manual").attach_store(store)
    strategy.channel("log")
    strategy.channel("custom", handler=lambda request: {"handled": request})

    asyncio.run(strategy.invoke_resource("log", {"op": "append", "payload": 2}))
    custom = asyncio.run(strategy.invoke_resource("custom", {"x": 1}))

    assert [event.payload for event in store.query_events("log")] == [2]
    assert store.get_channel("log").description == "Created by hand."
    assert custom == {"handled": {"x": 1}}
    assert [channel.name for channel in store.list_channels()] == ["log"]


def test_channel_without_a_store_still_echoes():
    strategy = Strategy("bare")
    strategy.channel("log")

    output = asyncio.run(strategy.invoke_resource("log", {"op": "append"}))

    assert output["resource"]["name"] == "log"
    assert output["input"] == {"op": "append"}


# SH2: one store per strategy, shared by every package; nothing is silently dropped.


def test_packages_share_the_strategy_store_served_at_network(tmp_path: Path):
    alpha = _package(tmp_path / "alpha", "alpha")
    beta = _package(tmp_path / "beta", "beta")
    strategy = Strategy("ops").use_package(alpha, prefix="a").use_package(beta, prefix="b")
    client = TestClient(create_strategy_app(strategy))

    client.post("/channels/a.events/events", json={"input": {"payload": "from alpha"}})
    client.post("/channels/b.events/events", json={"input": {"payload": "from beta"}})

    assert strategy.store.root == (alpha / ".lllm" / "network").resolve()
    assert not (beta / ".lllm").exists()
    names = sorted(channel["name"] for channel in client.get("/network/channels").json())
    assert names == ["a.events", "b.events"]
    for channel, payload in (("a.events", "from alpha"), ("b.events", "from beta")):
        events = client.get("/network/events", params={"channel": channel}).json()
        assert [event["payload"] for event in events] == [payload]
    channels = {resource["name"]: resource for resource in client.get("/channels").json()}
    assert channels["b.events"]["metadata"]["store"] == str(strategy.store.root)


def test_prefixed_packages_sharing_a_store_folder_keep_separate_channels(tmp_path: Path):
    shared = tmp_path / "shared"
    strategy = (
        Strategy("ops")
        .use_package(_package(tmp_path / "alpha", "alpha"), prefix="a", store=shared)
        .use_package(_package(tmp_path / "beta", "beta"), prefix="b", store=shared)
    )
    client = TestClient(create_strategy_app(strategy))

    client.post("/channels/a.events/events", json={"input": {"payload": "only alpha"}})

    assert [e["payload"] for e in client.get("/channels/a.events/events").json()["output"]] == [
        "only alpha"
    ]
    assert client.get("/channels/b.events/events").json()["output"] == []
    assert strategy.store.root == shared.resolve()


def test_clashing_package_names_are_rejected_before_anything_is_added(tmp_path: Path):
    alpha = _package(tmp_path / "alpha", "alpha")
    beta = _package(tmp_path / "beta", "beta")
    strategy = Strategy("ops").use_package(alpha)
    before = strategy.resources

    with pytest.raises(ValueError, match="echo, events. Mount it with a prefix"):
        strategy.use_package(beta)

    assert strategy.resources == before
    strategy.use_package(beta, prefix="b")
    assert {"b.beta", "b.echo", "b.events"} <= {resource.name for resource in strategy.resources}


def test_a_strategy_never_swaps_its_store_silently(tmp_path: Path):
    alpha = _package(tmp_path / "alpha", "alpha")
    beta = _package(tmp_path / "beta", "beta")
    alpha_store = alpha / ".lllm" / "network"
    strategy = Strategy("ops").use_package(alpha, prefix="a")

    with pytest.raises(ValueError, match="a strategy has one store"):
        strategy.use_package(beta, prefix="b", store=tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="a strategy has one store"):
        strategy.attach_store(LocalStore(tmp_path / "other"))

    assert not any(resource.name.startswith("b.") for resource in strategy.resources)
    assert not (tmp_path / "elsewhere").exists()
    # The same folder is the same store, whether passed as a path or a store.
    strategy.use_package(beta, prefix="b", store=alpha_store)
    strategy.attach_store(LocalStore(alpha_store))
    assert strategy.store.root == alpha_store.resolve()


def test_unbound_package_channels_stay_metadata_only(tmp_path: Path):
    store = LocalStore(tmp_path / "store")
    strategy = Strategy("meta").attach_store(store)
    strategy.use_package(_package(tmp_path / "pkg"), bind=False)

    output = asyncio.run(strategy.invoke_resource("events", {"op": "append", "payload": 1}))
    strategy.store.open()

    assert output["resource"]["name"] == "events"
    assert store.list_channels() == ()


# SH3: shell routes use the services error envelope with meaningful status codes.


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ChannelNotFoundError("Channel not found: gone"), 404),
        (ChannelExistsError("Channel already exists: gone"), 409),
        (InvalidPayloadError("Invalid event."), 400),
    ],
)
def test_channel_handler_network_errors_keep_store_status_codes(error, status):
    def handler(input_value=None, *, context=None):
        raise error

    strategy = Strategy("errs")
    strategy.channel("gone", handler=handler)
    client = TestClient(create_strategy_app(strategy))

    response = client.get("/channels/gone/events")

    assert response.status_code == status
    assert _error(response)["type"] == type(error).__name__
    assert _error(response)["message"] == str(error)
    assert _error(response)["endpoint"] == "channel"
    assert _error(response)["metadata"] == {"resource": "gone"}


def test_unknown_resource_is_a_clean_404():
    client = TestClient(create_strategy_app(Strategy("empty")))

    response = client.post("/resources/nope/invoke", json={"input": 1})

    assert response.status_code == 404
    assert _error(response)["type"] == "ResourceNotFoundError"
    assert _error(response)["message"] == "Strategy resource not found: nope"
    with pytest.raises(KeyError):
        asyncio.run(Strategy("empty").invoke_resource("nope"))
    assert str(ResourceNotFoundError("Strategy resource not found: x")) == (
        "Strategy resource not found: x"
    )


def test_invalid_input_is_a_400_envelope(tmp_path: Path):
    client = TestClient(create_strategy_app(load_strategy(_package(tmp_path / "pkg"))))

    store_error = client.post(
        "/channels/events/events",
        json={"input": {"payload": 1, "kind": "not a token"}},
    )
    handler_error = client.get("/channels/events/events", params={"limit": 0})
    body_error = client.post("/resources/events/invoke", json={"context": "nope"})
    query_error = client.get("/channels/events/events", params={"limit": "many"})

    assert store_error.status_code == 400
    assert _error(store_error)["type"] == "InvalidPayloadError"
    assert handler_error.status_code == 400
    assert _error(handler_error) == {
        "type": "ValueError",
        "message": "limit must be a positive integer",
        "tactic": None,
        "endpoint": "channel",
        "request_id": None,
        "metadata": {"resource": "events"},
    }
    assert body_error.status_code == 400
    assert _error(body_error)["type"] == "SchemaError"
    assert _error(body_error)["message"] == "context: Input should be a valid dictionary"
    assert query_error.status_code == 400
    assert _error(query_error)["message"].startswith("query.limit: ")


def test_unexpected_handler_errors_are_500_envelopes_not_404():
    strategy = Strategy("keys")
    strategy.service("lookup", handler=lambda payload: payload["missing"])

    @strategy.runner
    def run(input_value):
        raise RuntimeError("runner broke")

    client = TestClient(create_strategy_app(strategy))

    lookup = client.post("/resources/lookup/invoke", json={"input": {}})
    run_response = client.post("/run", json={"input": 1})

    assert lookup.status_code == 500
    assert _error(lookup)["type"] == "KeyError"
    assert run_response.status_code == 500
    assert _error(run_response)["message"] == "runner broke"
    assert _error(run_response)["endpoint"] == "run"


def test_shell_errors_have_the_tactic_service_envelope_shape():
    class Broken(Tactic[Any, Any]):
        name = "broken"

        def _run(self, input_value, *, context=None):
            raise RuntimeError("tactic broke")

    strategy = Strategy("shape").add_tactic(Broken())
    client = TestClient(create_strategy_app(strategy))

    service_error = client.post("/tactics/broken/run", json={"input": 1})
    shell_error = client.post("/resources/broken/invoke", json={"input": 1})
    missing = client.post("/resources/missing/invoke", json={"input": 1})

    assert service_error.status_code == shell_error.status_code == 500
    assert set(_error(shell_error)) == set(_error(service_error))
    assert set(_error(missing)) == set(_error(service_error))
    assert _error(shell_error)["message"] == "tactic broke"


def test_handler_http_exceptions_keep_their_status():
    def handler(input_value=None):
        raise HTTPException(status_code=403, detail="no")

    strategy = Strategy("http")
    strategy.service("guarded", handler=handler)
    client = TestClient(create_strategy_app(strategy))

    response = client.post("/resources/guarded/invoke", json={"input": 1})

    assert response.status_code == 403
    assert response.json() == {"detail": "no"}


def test_error_mapping_does_not_import_the_network_module(monkeypatch):
    def handler(input_value=None):
        raise ValueError("bad input")

    strategy = Strategy("plain")
    strategy.service("check", handler=handler)
    client = TestClient(create_strategy_app(strategy))
    real_import = builtins.__import__

    def blocked_network_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = name
        if level and globals and globals.get("__package__"):
            package = globals["__package__"]
            base = package.rsplit(".", level - 1)[0] if level > 1 else package
            resolved = f"{base}.{name}" if name else base
        if resolved == "lllm.network" or resolved.startswith("lllm.network."):
            raise ImportError(f"blocked network module: {resolved}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_network_import)
    response = client.post("/resources/check/invoke", json={"input": 1})

    assert response.status_code == 400
    assert _error(response)["message"] == "bad input"


# SH4: tactic resources run through the tactic path on /resources/{name}/invoke.


def test_plain_function_tactic_gets_the_same_call_context_on_both_routes():
    def probe(input_value, *, context=None):
        return {
            "context_type": type(context).__name__,
            "metadata": context.metadata,
            "endpoint": context.endpoint,
            "request_id": context.request_id,
        }

    strategy = Strategy("ctx")
    strategy.tactic("probe", handler=probe)
    client = TestClient(create_strategy_app(strategy))
    body = {
        "input": {"x": 1},
        "context": {"metadata": {"request": "r1"}, "request_id": "req-1"},
    }

    via_tactic = client.post("/tactics/probe/run", json=body).json()["output"]
    via_resource = client.post("/resources/probe/invoke", json=body).json()["output"]

    assert via_resource == via_tactic == {
        "context_type": "CallContext",
        "metadata": {"request": "r1"},
        "endpoint": "run",
        "request_id": "req-1",
    }


def test_tactic_objects_and_package_tactics_get_the_request_call_context(tmp_path: Path):
    class Probe(Tactic[Any, Any]):
        name = "probe"

        def _run(self, input_value, *, context=None):
            return {"metadata": context.metadata, "tags": context.tags}

    strategy = Strategy("ctx").add_tactic(Probe())
    strategy.use_package(_package(tmp_path / "pkg"), prefix="pkg")
    client = TestClient(create_strategy_app(strategy))
    body = {"input": 1, "context": {"metadata": {"k": "v"}, "tags": {"team": "a"}}}

    probe = client.post("/resources/probe/invoke", json=body).json()["output"]
    echo = client.post("/resources/pkg.echo/invoke", json=body).json()["output"]

    assert probe == {"metadata": {"k": "v"}, "tags": {"team": "a"}}
    assert echo["context_type"] == "CallContext"
    assert echo["metadata"] == {"k": "v"}


def test_non_tactic_resources_keep_the_dict_context():
    seen: list[Any] = []

    def handler(input_value, context=None):
        seen.append(context)
        return input_value

    class Metadata(Tactic[Any, Any]):
        name = "metadata"

        def _run(self, input_value, *, context=None):
            return context.metadata

    strategy = Strategy("ctx")
    strategy.service("svc", handler=handler)
    strategy.service("wrapped", handler=Metadata())
    client = TestClient(create_strategy_app(strategy))
    body = {"input": 1, "context": {"request": "r1"}}

    client.post("/resources/svc/invoke", json=body)
    wrapped = client.post("/resources/wrapped/invoke", json=body).json()["output"]

    assert seen == [{"request": "r1"}]
    assert wrapped == {"request": "r1"}


def test_invoke_resource_accepts_a_call_context_from_python():
    strategy = Strategy("ctx")
    strategy.tactic("probe", handler=lambda value, context=None: context.request_id)

    output = asyncio.run(
        strategy.invoke_resource("probe", 1, context=CallContext(request_id="py-1"))
    )

    assert output == "py-1"


# SH5: package tactics carry their manifest ref.


def test_package_tactics_report_the_manifest_ref(tmp_path: Path):
    ref = "lllm://demo/analysis-pack/tactics/echo"
    strategy = load_strategy(_package(tmp_path / "pkg"))
    prefixed = Strategy("ops").use_package(_package(tmp_path / "pkg2"), prefix="p")
    client = TestClient(create_strategy_app(strategy))
    prefixed_client = TestClient(create_strategy_app(prefixed))

    assert client.get("/tactics/echo/info").json()["package_ref"] == ref
    assert prefixed_client.get("/tactics/p.echo/info").json()["package_ref"] == ref
    assert strategy.tactics()["echo"].package_ref == ref
    invoked = client.post("/resources/echo/invoke", json={"input": 1}).json()["output"]
    assert invoked["tactic_ref"] == ref


# SH6: building or inspecting a strategy never creates its store.


def test_inspecting_a_package_never_creates_its_store(tmp_path: Path, capsys):
    package = _package(tmp_path / "pkg")

    assert main(["inspect", str(package)]) == 0
    assert main(["inspect", str(package), "--json"]) == 0

    assert "channel events lllm://demo/analysis-pack/channels/events" in capsys.readouterr().out
    assert not (package / ".lllm").exists()


def test_building_a_strategy_defers_the_store_until_a_channel_is_used(tmp_path: Path):
    package = _package(tmp_path / "pkg")
    strategy = load_strategy(package)
    TacticProcessor(strategy.tactics()["echo"], strategy.store, source="events")

    channel = next(resource for resource in strategy.resources if resource.kind == "channel")

    assert isinstance(strategy.store, StrategyStore)
    assert strategy.store.root == (package / ".lllm" / "network").resolve()
    assert channel.metadata["store"] == str(strategy.store.root)
    assert not (package / ".lllm").exists()

    asyncio.run(strategy.invoke_resource("events", {"op": "append", "payload": 1}))

    assert (package / ".lllm" / "network").is_dir()
    assert [event.payload for event in strategy.store.query_events("events")] == [1]


def test_serving_opens_the_store_with_its_declared_channels(tmp_path: Path):
    package = _package(tmp_path / "pkg", channels=("events", "alerts"))
    strategy = load_strategy(package)

    client = TestClient(create_strategy_app(strategy))

    assert (package / ".lllm" / "network").is_dir()
    channels = {channel["name"]: channel for channel in client.get("/network/channels").json()}
    assert sorted(channels) == ["alerts", "events"]
    assert channels["alerts"]["description"] == "The alerts stream."
