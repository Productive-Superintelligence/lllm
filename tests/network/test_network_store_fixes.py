"""Regression tests for network store, client, server, resolver, and hub fixes."""

import asyncio
import importlib.util
import inspect
import json
import math
import time
from pathlib import Path
from typing import get_args

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import lllm.network as network
import lllm.network.core as network_core
from lllm import Resolver
from lllm.hub import ChannelResource, PackageManifest
from lllm.hub.models import CHANNEL_FORM_VALUES
from lllm.network import (
    AsyncRemoteStore,
    ChannelForm,
    ChannelRefError,
    ChannelResolver,
    EventExistsError,
    InvalidPayloadError,
    LocalStore,
    NetworkError,
    RemoteStore,
    RemoteStoreError,
    Snapshot,
    SnapshotNotFoundError,
    create_app,
    endpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CHANNEL_REF = "lllm://demo/combo/channels/events"
OTHER_REF = "lllm://demo/combo/channels/other"
SNAPSHOT_REF = "lllm://demo/combo/snapshots/latest"


def request(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def sync_transport(app) -> httpx.MockTransport:
    """Route the synchronous RemoteStore into an in-process ASGI app."""

    test_client = TestClient(app)

    def handler(outgoing: httpx.Request) -> httpx.Response:
        response = test_client.request(
            outgoing.method,
            outgoing.url.raw_path.decode("ascii"),
            content=outgoing.content,
            headers={"content-type": outgoing.headers.get("content-type", "")},
        )
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers={"content-type": response.headers.get("content-type", "")},
        )

    return httpx.MockTransport(handler)


class Blocking:
    """Call any store's methods synchronously, awaiting coroutine results."""

    def __init__(self, store):
        self._store = store

    def __getattr__(self, name):
        method = getattr(self._store, name)

        def call(*args, **kwargs):
            result = method(*args, **kwargs)
            return asyncio.run(result) if inspect.isawaitable(result) else result

        return call


def make_store(kind: str, root: Path) -> Blocking:
    store = LocalStore(root)
    if kind == "local":
        return Blocking(store)
    app = create_app(store)
    if kind == "remote":
        return Blocking(RemoteStore("http://testserver", transport=sync_transport(app)))
    return Blocking(
        AsyncRemoteStore("http://testserver", transport=httpx.ASGITransport(app=app))
    )


@pytest.fixture(params=["local", "remote", "async"])
def any_store(request, tmp_path):
    return make_store(request.param, tmp_path / "store")


@pytest.fixture(params=["remote", "async"])
def remote_store(request, tmp_path):
    return make_store(request.param, tmp_path / "store")


def load_example(*parts: str):
    path = ROOT.joinpath("examples", "network", *parts)
    spec = importlib.util.spec_from_file_location(f"fixes_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# S-N1: duplicate event ids


def test_event_exists_error_is_exported_like_other_network_errors():
    assert issubclass(EventExistsError, NetworkError)
    assert network.EventExistsError is network_core.EventExistsError
    assert "EventExistsError" in network.__all__
    assert "EventExistsError" in network_core.__all__


def test_local_store_rejects_duplicate_event_ids(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    store.create_channel({"name": "other"})
    original = store.append_event({"id": "e1", "channel": "events", "payload": 1})

    for channel in ("events", "other"):
        with pytest.raises(EventExistsError, match="Event already exists: e1"):
            store.append_event({"id": "e1", "channel": channel, "payload": 2})

    assert store.get_event("e1") == original
    assert store.query_events("events") == (original,)
    assert store.query_events("other") == ()


def test_local_store_does_not_report_other_insert_failures_as_duplicates(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})

    with pytest.raises(Exception) as exc_info:
        store.append_event({"id": "e1", "channel": "events", "timestamp": math.nan})

    assert not isinstance(exc_info.value, EventExistsError)


def test_server_returns_409_for_duplicate_event_ids(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))
    request(app, "POST", "/channels", json={"name": "events"})
    first = request(app, "POST", "/events", json={"id": "e1", "channel": "events"})
    duplicate = request(app, "POST", "/events", json={"id": "e1", "channel": "events"})

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.json() == {
        "detail": {
            "error": {"type": "EventExistsError", "message": "Event already exists: e1"}
        }
    }


def test_remote_clients_raise_clear_error_for_duplicate_event_ids(remote_store):
    remote_store.create_channel({"name": "events"})
    remote_store.append_event({"id": "e1", "channel": "events"})

    with pytest.raises(RemoteStoreError) as exc_info:
        remote_store.append_event({"id": "e1", "channel": "events"})

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_type == "EventExistsError"
    assert exc_info.value.message == "Event already exists: e1"


# S-N2: snapshot values are stored exactly as given


@pytest.mark.parametrize(
    "value",
    [
        {"value": 3, "unit": "ms"},
        {"value": 3},
        {"timestamp": 12.5},
        {"metadata": {"a": 1}, "x": 2},
        {"name": "other", "channel": "missing", "value": None},
    ],
)
def test_put_snapshot_stores_mapping_values_exactly(any_store, value):
    snapshot = any_store.put_snapshot("config", value)

    assert snapshot.name == "config"
    assert snapshot.value == value
    assert snapshot.channel is None
    assert snapshot.metadata == {}
    assert any_store.get_snapshot("config").value == value


@pytest.mark.parametrize("client_cls", [RemoteStore, AsyncRemoteStore])
def test_remote_put_snapshot_sends_mapping_values_as_the_value(client_cls):
    bodies = []

    def handler(outgoing: httpx.Request) -> httpx.Response:
        assert outgoing.url.path == "/snapshots/config"
        body = json.loads(outgoing.content.decode("utf-8"))
        bodies.append(body)
        return httpx.Response(200, json={"name": "config", "timestamp": 1.0, **body})

    transport = httpx.MockTransport(handler)
    client = Blocking(client_cls("http://testserver", transport=transport))
    snapshot = client.put_snapshot("config", {"value": 3, "unit": "ms"})

    assert bodies == [{"value": {"value": 3, "unit": "ms"}}]
    assert snapshot.value == {"value": 3, "unit": "ms"}


# S-N3: every store accepts the same put_snapshot forms


def test_put_snapshot_accepts_the_same_forms_on_every_store(any_store):
    any_store.create_channel({"name": "events"})
    event = any_store.append_event({"channel": "events", "payload": {"n": 0}})

    written = {
        "model": any_store.put_snapshot(
            Snapshot(name="model", channel="events", value={"n": 1})
        ),
        "mapping": any_store.put_snapshot(
            {"name": "mapping", "channel": "events", "value": {"n": 2}}
        ),
        "named": any_store.put_snapshot(
            "named",
            {"n": 3},
            channel="events",
            schema="demo:State",
            source_event_id=event.id,
            metadata={"role": "named"},
        ),
        "named-model": any_store.put_snapshot(
            "named-model",
            Snapshot(name="ignored", channel="events", value={"n": 4}),
            metadata={"role": "override"},
        ),
        "overridden": any_store.put_snapshot(
            {"name": "overridden", "value": {"n": 0}, "metadata": None},
            {"n": 5},
            channel="events",
        ),
        "none": any_store.put_snapshot("none"),
    }

    for name, snapshot in written.items():
        assert snapshot.name == name
        assert any_store.get_snapshot(name) == snapshot
    assert [written[name].value for name in written] == [
        {"n": 1},
        {"n": 2},
        {"n": 3},
        {"n": 4},
        {"n": 5},
        None,
    ]
    assert {written[name].channel for name in written if name != "none"} == {"events"}
    assert written["named"].schema == "demo:State"
    assert written["named"].source_event_id == event.id
    assert written["named"].metadata == {"role": "named"}
    assert written["named-model"].metadata == {"role": "override"}
    assert written["overridden"].metadata == {}


@pytest.mark.parametrize(
    "args",
    [
        ("bad/name", 1),
        ({"name": "bad/name", "value": 1},),
        ({"name": "bad name"},),
        ({"value": 1},),
        (None,),
    ],
)
def test_put_snapshot_rejects_invalid_names_on_every_store(any_store, args):
    with pytest.raises(InvalidPayloadError):
        any_store.put_snapshot(*args)


def test_put_snapshot_validates_fields_on_every_store(any_store):
    with pytest.raises(InvalidPayloadError, match="snapshot.metadata"):
        any_store.put_snapshot("latest", {}, metadata=[])
    with pytest.raises(InvalidPayloadError, match="snapshot.metadata"):
        any_store.put_snapshot({"name": "latest", "value": {}, "metadata": []})
    with pytest.raises(InvalidPayloadError):
        any_store.put_snapshot(Snapshot(name="latest"), channel="bad/name")


def test_put_snapshot_has_the_same_signature_on_every_store():
    signature = inspect.signature(LocalStore.put_snapshot)

    assert inspect.signature(RemoteStore.put_snapshot) == signature
    assert inspect.signature(AsyncRemoteStore.put_snapshot) == signature
    assert {"snapshot", "value", "name"} <= set(signature.parameters)


def test_put_snapshot_accepts_name_and_snapshot_keywords_on_every_store(any_store):
    written = [
        any_store.put_snapshot(name="by-name", value={"n": 1}),
        any_store.put_snapshot(snapshot=Snapshot(name="by-model", value={"n": 2})),
        any_store.put_snapshot(snapshot={"name": "by-dict", "value": {"n": 3}}),
        any_store.put_snapshot(
            snapshot=Snapshot(name="ignored", value={"n": 4}),
            name="renamed-model",
        ),
        any_store.put_snapshot(
            {"name": "ignored", "value": {"n": 5}},
            name="renamed-dict",
        ),
        any_store.put_snapshot(
            name="named-model",
            value=Snapshot(name="ignored", value={"n": 6}, metadata={"m": 6}),
        ),
    ]

    assert [(snapshot.name, snapshot.value) for snapshot in written] == [
        ("by-name", {"n": 1}),
        ("by-model", {"n": 2}),
        ("by-dict", {"n": 3}),
        ("renamed-model", {"n": 4}),
        ("renamed-dict", {"n": 5}),
        ("named-model", {"n": 6}),
    ]
    assert written[-1].metadata == {"m": 6}
    for snapshot in written:
        assert any_store.get_snapshot(snapshot.name) == snapshot
    with pytest.raises((SnapshotNotFoundError, RemoteStoreError)):
        any_store.get_snapshot("ignored")


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("latest", 1), {"name": "other"}),
        ((), {"snapshot": "latest", "name": "other"}),
    ],
)
def test_put_snapshot_rejects_a_name_given_twice_on_every_store(any_store, args, kwargs):
    with pytest.raises(TypeError, match="multiple values for argument 'name'"):
        any_store.put_snapshot(*args, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"value": 1},
        {"name": "bad/name", "value": 1},
        {"snapshot": {"value": 1}},
        {"value": Snapshot(name="unnamed", value=1)},
    ],
)
def test_put_snapshot_rejects_missing_or_invalid_keyword_names_on_every_store(
    any_store, kwargs
):
    with pytest.raises(InvalidPayloadError):
        any_store.put_snapshot(**kwargs)


def test_put_snapshot_treats_none_fields_as_unset_on_every_store(any_store):
    snapshot = any_store.put_snapshot(
        {
            "name": "unset",
            "value": 1,
            "channel": None,
            "timestamp": None,
            "schema": None,
            "source_event_id": None,
            "metadata": None,
        }
    )
    renamed = any_store.put_snapshot({"name": None, "value": 2}, name="renamed")

    assert snapshot.timestamp == pytest.approx(time.time(), abs=60)
    assert [snapshot.channel, snapshot.schema, snapshot.source_event_id] == [None] * 3
    assert snapshot.metadata == {}
    assert any_store.get_snapshot("unset") == snapshot
    assert (renamed.name, renamed.value) == ("renamed", 2)


# S-N4: from_text parses config in memory


def test_from_text_never_writes_config_files(tmp_path, monkeypatch):
    text = f"""
[refs."{CHANNEL_REF}"]
store = "data/store"
""".lstrip()
    fresh = tmp_path / "fresh"

    resolver = ChannelResolver.from_text(text, root=fresh)

    assert not fresh.exists()
    assert resolver.refs() == (CHANNEL_REF,)
    assert resolver.store_path(CHANNEL_REF) == (fresh / "data" / "store").resolve()

    existing = tmp_path / "existing" / ".lllm" / "config.toml"
    existing.parent.mkdir(parents=True)
    existing.write_text("# hand-written config\n", encoding="utf-8")

    ChannelResolver.from_text(text, root=tmp_path / "existing")

    assert existing.read_text(encoding="utf-8") == "# hand-written config\n"

    monkeypatch.chdir(tmp_path)
    resolver = ChannelResolver.from_text(text)

    assert resolver.root == tmp_path.resolve()
    assert not (tmp_path / ".lllm").exists()


# S-N5: store bindings can name a [stores.<name>] table


def test_store_binding_uses_named_store_table_path(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{CHANNEL_REF}"]
store = "default"

[refs."{OTHER_REF}"]
store = "data/plain"

[refs."{SNAPSHOT_REF}"]
path = "default"

[stores.default]
path = "data/shared"
""".lstrip(),
        root=tmp_path,
    )

    shared = (tmp_path / "data" / "shared").resolve()
    assert resolver.resolve(CHANNEL_REF).store == "default"
    assert resolver.store_path(CHANNEL_REF) == shared
    assert resolver.local_store(CHANNEL_REF).root == shared
    assert resolver.store_path(OTHER_REF) == (tmp_path / "data" / "plain").resolve()
    assert resolver.store_path(SNAPSHOT_REF) == (tmp_path / "default").resolve()


def test_store_binding_without_matching_table_stays_a_path(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{CHANNEL_REF}"]
store = "default"
""".lstrip(),
        root=tmp_path,
    )

    assert resolver.store_path(CHANNEL_REF) == (tmp_path / "default").resolve()


def test_named_store_table_paths_resolve_against_config_root(tmp_path):
    workspace = tmp_path / "workspace"
    config = workspace / ".lllm" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
[refs."{CHANNEL_REF}"]
store = "default"

[stores.default]
path = ".lllm/network"
""".lstrip(),
        encoding="utf-8",
    )
    expected = (workspace / ".lllm" / "network").resolve()

    assert ChannelResolver.from_config(config).store_path(CHANNEL_REF) == expected
    assert Resolver.from_config(workspace).store(CHANNEL_REF).root == expected


def test_named_store_table_without_path_is_rejected(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{CHANNEL_REF}"]
store = "remote"

[stores.remote]
owner = "ops"
""".lstrip(),
        root=tmp_path,
    )

    with pytest.raises(ChannelRefError, match=r"\[stores\.remote\] has no path"):
        resolver.store_path(CHANNEL_REF)


# S-N6: custom endpoints receive query parameters


def test_custom_endpoints_receive_query_parameters(tmp_path):
    @endpoint.get("/channels/{name}/echo", scope="channel")
    def echo(store, name: str, params=None, limit: int = 20):
        return {"name": name, "params": params, "limit": limit}

    @endpoint.post("/channels/{name}/echo", scope="channel")
    def echo_body(store, name: str, body=None, params=None):
        return {"body": body, "params": params}

    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    app = create_app(store, custom_endpoints=[echo, echo_body])

    queried = request(app, "GET", "/channels/events/echo?limit=5&tag=a&tag=b")
    bare = request(app, "GET", "/channels/events/echo")
    posted = request(app, "POST", "/channels/events/echo?dry_run=1", json={"x": 1})

    assert queried.json() == {
        "name": "events",
        "params": {"limit": "5", "tag": "b"},
        "limit": 20,
    }
    assert bare.json()["params"] == {}
    assert posted.json() == {"body": {"x": 1}, "params": {"dry_run": "1"}}


def test_custom_endpoint_path_parameter_named_params_binds_the_path_value(tmp_path):
    @endpoint.get("/things/{params}", scope="store")
    def thing(params: str):
        return {"params": params}

    app = create_app(LocalStore(tmp_path / "store"), custom_endpoints=[thing])
    response = request(app, "GET", "/things/hello?x=1")

    assert response.status_code == 200
    assert response.json() == {"params": "hello"}


def test_hub_manifest_tail_endpoint_returns_the_most_recent_events(tmp_path):
    module = load_example("hub_manifest", "package_manifest.py")
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "raw"})
    store.create_channel({"name": "analysis"})
    for n in range(1, 31):
        store.append_event({"channel": "raw", "payload": {"n": n}})
        store.append_event({"channel": "analysis", "payload": {"n": -n}})
    app = create_app(store, custom_endpoints=[module.channel_tail])

    default = request(app, "GET", "/channels/raw/tail")
    limited = request(app, "GET", "/channels/raw/tail", params={"limit": "3"})
    oversized = request(app, "GET", "/channels/raw/tail", params={"limit": "500"})
    invalid = [
        request(app, "GET", "/channels/raw/tail", params={"limit": value})
        for value in ("0", "-1", "ten")
    ]

    assert [event["payload"]["n"] for event in default.json()] == list(range(11, 31))
    assert [event["payload"]["n"] for event in limited.json()] == [28, 29, 30]
    assert [event["payload"]["n"] for event in oversized.json()] == list(range(1, 31))
    for response in invalid:
        assert response.status_code == 400
        assert response.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert "most recent" in module.channel_tail.__lllm_network_endpoint__.description


def test_hub_manifest_tail_endpoint_rejects_out_of_range_limits(tmp_path):
    module = load_example("hub_manifest", "package_manifest.py")
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "raw"})
    store.append_event({"channel": "raw", "payload": {"n": 1}})
    app = create_app(store, custom_endpoints=[module.channel_tail])

    def tail(limit: str):
        return request(app, "GET", "/channels/raw/tail", params={"limit": limit})

    assert [event["payload"] for event in tail("1000").json()] == [{"n": 1}]
    for limit in ("1001", "99999999999999999999", "9" * 5000, "", "2.5"):
        response = tail(limit)
        assert response.status_code == 400, limit[:30]
        assert response.json()["detail"]["error"] == {
            "type": "InvalidPayloadError",
            "message": "limit must be an integer from 1 to 1000.",
        }


# S-N7: hub channel forms match the network's


def test_hub_channel_forms_mirror_network_channel_forms():
    assert CHANNEL_FORM_VALUES == frozenset(get_args(ChannelForm))
    for form in get_args(ChannelForm):
        assert ChannelResource(form=form).form == form


@pytest.mark.parametrize("form", ["stream", "Log", "logs"])
def test_hub_channel_resource_rejects_forms_the_network_cannot_create(form):
    with pytest.raises(ValidationError, match="channel.form must be one of"):
        ChannelResource(form=form)
    with pytest.raises(ValidationError, match="channel.form must be one of"):
        PackageManifest.model_validate(
            {"package": {"name": "demo"}, "channels": {"events": {"form": form}}}
        )
