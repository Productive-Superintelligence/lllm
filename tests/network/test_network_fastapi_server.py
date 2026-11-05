import asyncio
from types import MappingProxyType

import httpx
import pytest
from pydantic import ValidationError

from lllm.network import LocalStore
from lllm.network.server import create_app, endpoint
from lllm.network.server.endpoints import StoreEndpointSpec, endpoint_spec
from lllm.network.server.fastapi import (
    ArtifactWriteRequest,
    ErrorDetail,
    SnapshotWriteRequest,
    SubscriptionRequest,
)


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_fastapi_request_models_isolate_mutable_constructor_inputs():
    filters = {"kind": ["raw"]}
    subscription_metadata = {"tags": ["sub"]}
    subscription = SubscriptionRequest(
        channel="events",
        filters=filters,
        metadata=subscription_metadata,
    )
    filters["kind"].append("mutated")
    subscription_metadata["tags"].append("mutated")

    artifact_metadata = {"tags": ["artifact"]}
    artifact = ArtifactWriteRequest(data="hello", metadata=artifact_metadata)
    artifact_metadata["tags"].append("mutated")

    snapshot_value = {"state": ["ok"]}
    snapshot_metadata = {"tags": ["snapshot"]}
    snapshot = SnapshotWriteRequest(
        value=snapshot_value,
        metadata=snapshot_metadata,
    )
    snapshot_value["state"].append("mutated")
    snapshot_metadata["tags"].append("mutated")

    assert subscription.filters == {"kind": ["raw"]}
    assert subscription.metadata == {"tags": ["sub"]}
    assert artifact.metadata == {"tags": ["artifact"]}
    assert snapshot.value == {"state": ["ok"]}
    assert snapshot.metadata == {"tags": ["snapshot"]}


def test_fastapi_request_models_accept_nested_read_only_mapping_inputs():
    filter_inner = {"kind": "raw"}
    subscription_inner = {"tag": "sub"}
    subscription = SubscriptionRequest(
        channel="events",
        filters={"nested": MappingProxyType(filter_inner)},
        metadata={"nested": MappingProxyType(subscription_inner)},
    )
    artifact_inner = {"tag": "artifact"}
    artifact = ArtifactWriteRequest(
        data="hello",
        metadata={"nested": MappingProxyType(artifact_inner)},
    )
    value_inner = {"state": "ok"}
    snapshot_inner = {"tag": "snapshot"}
    snapshot = SnapshotWriteRequest(
        value={"nested": MappingProxyType(value_inner)},
        metadata={"nested": MappingProxyType(snapshot_inner)},
    )

    filter_inner["kind"] = "changed"
    subscription_inner["tag"] = "changed"
    artifact_inner["tag"] = "changed"
    value_inner["state"] = "changed"
    snapshot_inner["tag"] = "changed"

    assert subscription.filters == {"nested": {"kind": "raw"}}
    assert subscription.metadata == {"nested": {"tag": "sub"}}
    assert artifact.metadata == {"nested": {"tag": "artifact"}}
    assert snapshot.value == {"nested": {"state": "ok"}}
    assert snapshot.metadata == {"nested": {"tag": "snapshot"}}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SubscriptionRequest(channel=b"events"),
        lambda: SubscriptionRequest(id=b"sub", channel="events"),
        lambda: SubscriptionRequest(channel="events", consumer=b"worker"),
        lambda: ArtifactWriteRequest(data=b"hello"),
        lambda: ArtifactWriteRequest(data="hello", channel=b"events"),
        lambda: ArtifactWriteRequest(data="hello", media_type=b"text/plain"),
        lambda: ArtifactWriteRequest(data="hello", event_ids=(b"event",)),
        lambda: SnapshotWriteRequest(channel=b"state"),
        lambda: SnapshotWriteRequest(schema=b"demo.schemas:State"),
        lambda: SnapshotWriteRequest(source_event_id=b"event"),
    ],
)
def test_fastapi_request_models_reject_bytes_for_string_fields(factory):
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.parametrize(
    "value",
    (
        "",
        "   ",
        ".",
        "..",
        "bad type",
        "bad/type",
        "bad:type",
        "bad\\type",
        "bad%2Ftype",
    ),
)
def test_fastapi_error_detail_rejects_malformed_type_tokens(value):
    with pytest.raises(ValidationError):
        ErrorDetail(type=value, message="bad")


def test_fastapi_error_detail_allows_common_type_tokens():
    error = ErrorDetail(type="InvalidPayloadError", message="bad")

    assert error.type == "InvalidPayloadError"


def test_endpoint_decorator_normalizes_relative_paths():
    @endpoint.get(
        "channels/{name}/count",
        name="count_events",
        scope="channel",
        tags=("events",),
    )
    def count_events(store: LocalStore, name: str):
        return {"count": len(store.query_events(name))}

    spec = endpoint_spec(count_events)

    assert spec is not None
    assert spec.path == "/channels/{name}/count"
    assert spec.name == "count_events"
    assert spec.scope == "channel"
    assert spec.tags == ("events",)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: endpoint.get(None),
        lambda: endpoint.get(123),
        lambda: endpoint.get(""),
        lambda: endpoint.get("bad path"),
        lambda: endpoint.get("/channels%2Fevents"),
        lambda: endpoint.get("/channels?name=events"),
        lambda: endpoint.get("/channels#events"),
        lambda: endpoint.get("http://example.com/channels"),
        lambda: endpoint.get("//example.com/channels"),
        lambda: endpoint.get("/channels", name=""),
        lambda: endpoint.get("/channels", name=123),
        lambda: endpoint.get("/channels", name="bad name"),
        lambda: endpoint.get("/channels", name="bad%2Fname"),
        lambda: endpoint.get("/channels", scope="dataset"),
        lambda: endpoint.get("/channels", description=123),
        lambda: endpoint.get("/channels", tags="events"),
        lambda: endpoint.get("/channels", tags=(123,)),
        lambda: endpoint.get("/channels", tags=("",)),
        lambda: endpoint.get("/channels", tags=("bad tag",)),
        lambda: endpoint.get("/channels", tags=("bad%2Ftag",)),
        lambda: StoreEndpointSpec(method=" GET ", path="/channels", name="channels"),
        lambda: StoreEndpointSpec(method="TRACE", path="/channels", name="channels"),
    ],
)
def test_endpoint_decorator_rejects_malformed_metadata(factory):
    with pytest.raises(ValueError):
        factory()


def test_custom_endpoint_collection_rejects_duplicate_names_and_routes(tmp_path):
    @endpoint.get("/custom/first", name="shared")
    def first_custom():
        return {}

    @endpoint.post("/custom/second", name="shared")
    def second_custom():
        return {}

    with pytest.raises(ValueError, match="duplicate custom endpoint name"):
        create_app(
            LocalStore(tmp_path / "store-names"),
            custom_endpoints=[first_custom, second_custom],
        )

    @endpoint.get("/custom/{name}", name="by_name")
    def by_name(name: str):
        return {"name": name}

    @endpoint.get("/custom/{id}", name="by_id")
    def by_id(id: str):
        return {"id": id}

    with pytest.raises(ValueError, match="duplicate custom endpoint route"):
        create_app(
            LocalStore(tmp_path / "store-routes"),
            custom_endpoints=[by_name, by_id],
        )


def test_custom_endpoint_routes_cannot_shadow_service_routes(tmp_path):
    @endpoint.get("/health", name="custom_health")
    def custom_health():
        return {"ok": "custom"}

    with pytest.raises(ValueError, match="reserved LLLM network service route"):
        create_app(
            LocalStore(tmp_path / "store-health"),
            custom_endpoints=[custom_health],
        )

    @endpoint.get("/channels/{channel_name}", name="custom_channel")
    def custom_channel(channel_name: str):
        return {"name": channel_name}

    with pytest.raises(ValueError, match="reserved LLLM network service route"):
        create_app(
            LocalStore(tmp_path / "store-channel"),
            custom_endpoints=[custom_channel],
        )


def test_fastapi_channel_event_subscription_flow(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    health = request(app, "GET", "/health")
    created = request(
        app,
        "POST",
        "/channels",
        json={"name": "events", "schema": "demo.schemas:Event", "form": "log"},
    )
    duplicate = request(app, "POST", "/channels", json={"name": "events"})
    listed = request(app, "GET", "/channels")
    event = request(
        app,
        "POST",
        "/events",
        json={"channel": "events", "kind": "raw", "payload": {"n": 1}},
    )
    dangling_child = request(
        app,
        "POST",
        "/events",
        json={
            "channel": "events",
            "kind": "analysis",
            "payload": {"n": 2},
            "parent_ids": ["missing-event"],
        },
    )
    queried = request(app, "GET", "/events", params={"channel": "events"})
    loaded_event = request(app, "GET", f"/events/{event.json()['id']}")
    missing_event = request(app, "GET", "/events/missing")
    sub = request(app, "POST", "/subscriptions", json={"channel": "events"})
    request(app, "POST", "/channels", json={"name": "other-events"})
    conflicting_sub = request(
        app,
        "POST",
        "/subscriptions",
        json={"id": sub.json()["id"], "channel": "other-events"},
    )
    pulled = request(
        app,
        "POST",
        f"/subscriptions/{sub.json()['id']}/pull",
    )
    loaded_sub = request(app, "GET", f"/subscriptions/{sub.json()['id']}")
    pulled_again = request(
        app,
        "POST",
        f"/subscriptions/{sub.json()['id']}/pull",
    )

    assert health.json() == {"ok": True}
    assert created.status_code == 200
    assert created.json()["name"] == "events"
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["error"]["type"] == "ChannelExistsError"
    assert listed.json()[0]["name"] == "events"
    assert event.json()["payload"] == {"n": 1}
    assert dangling_child.status_code == 404
    assert dangling_child.json()["detail"]["error"]["type"] == "EventNotFoundError"
    assert queried.json()[0]["id"] == event.json()["id"]
    assert loaded_event.json()["id"] == event.json()["id"]
    assert missing_event.status_code == 404
    assert missing_event.json()["detail"]["error"]["type"] == "EventNotFoundError"
    assert conflicting_sub.status_code == 409
    assert conflicting_sub.json()["detail"]["error"]["type"] == "SubscriptionExistsError"
    assert [item["id"] for item in pulled.json()] == [event.json()["id"]]
    assert loaded_sub.json()["cursor"] == event.json()["cursor"]
    assert pulled_again.json() == []


def test_fastapi_returns_stable_errors_for_cursor_edges(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))
    request(app, "POST", "/channels", json={"name": "events"})
    sub = request(app, "POST", "/subscriptions", json={"channel": "events"})

    bad_cursor = request(
        app,
        "GET",
        "/events",
        params={"channel": "events", "after_cursor": -1},
    )
    bad_limit = request(
        app,
        "POST",
        f"/subscriptions/{sub.json()['id']}/pull",
        params={"limit": 0},
    )
    bad_batch = request(
        app,
        "POST",
        "/subscriptions",
        json={"channel": "events", "batch_size": 0},
    )
    bad_filter = request(
        app,
        "POST",
        "/subscriptions",
        json={"channel": "events", "filters": {"source": "sensor"}},
    )
    missing_channel = request(app, "GET", "/channels/missing")

    assert bad_cursor.status_code == 400
    assert bad_cursor.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert "after_cursor" in bad_cursor.json()["detail"]["error"]["message"]
    assert bad_limit.status_code == 400
    assert bad_limit.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert bad_batch.status_code == 400
    assert bad_batch.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert bad_filter.status_code == 400
    assert bad_filter.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert "unsupported" in bad_filter.json()["detail"]["error"]["message"]
    assert missing_channel.status_code == 404
    assert missing_channel.json()["detail"]["error"]["type"] == "ChannelNotFoundError"


def test_fastapi_returns_stable_errors_for_request_validation(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    bad_channel = request(app, "POST", "/channels", json={"name": "bad/name"})
    bad_channel_space = request(app, "POST", "/channels", json={"name": "bad name"})
    bad_channel_percent = request(
        app,
        "POST",
        "/channels",
        json={"name": "bad%2Fname"},
    )
    bad_event = request(
        app,
        "POST",
        "/events",
        json={"id": "bad/name", "channel": "events", "payload": {}},
    )
    missing_event_channel = request(app, "POST", "/events", json={"payload": {}})

    for response, field in (
        (bad_channel, "channel.name"),
        (bad_channel_space, "channel.name"),
        (bad_channel_percent, "channel.name"),
        (bad_event, "event.id"),
        (missing_event_channel, "channel"),
    ):
        assert response.status_code == 400
        assert response.json()["detail"]["error"]["type"] == "InvalidPayloadError"
        assert field in response.json()["detail"]["error"]["message"]


def test_fastapi_artifact_and_snapshot_flow(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))
    request(app, "POST", "/channels", json={"name": "state", "form": "latest-state"})

    artifact = request(
        app,
        "POST",
        "/artifacts",
        json={
            "data": "hello",
            "encoding": "text",
            "channel": "state",
            "media_type": "text/plain",
        },
    )
    dangling_artifact = request(
        app,
        "POST",
        "/artifacts",
        json={
            "data": "hello",
            "encoding": "text",
            "event_ids": ["missing-event"],
        },
    )
    invalid_base64 = request(
        app,
        "POST",
        "/artifacts",
        json={
            "data": "not base64!",
            "encoding": "base64",
            "channel": "state",
        },
    )
    artifact_data = request(app, "GET", f"/artifacts/{artifact.json()['id']}")
    artifact_metadata = request(
        app,
        "GET",
        f"/artifacts/{artifact.json()['id']}/metadata",
    )
    missing_artifact = request(app, "GET", "/artifacts/missing")
    missing_artifact_metadata = request(app, "GET", "/artifacts/missing/metadata")

    snapshot = request(
        app,
        "PUT",
        "/snapshots/latest",
        json={"channel": "state", "value": {"status": "ok"}},
    )
    dangling_snapshot = request(
        app,
        "PUT",
        "/snapshots/dangling",
        json={"value": {"status": "bad"}, "source_event_id": "missing-event"},
    )
    loaded = request(app, "GET", "/snapshots/latest")
    missing_snapshot = request(app, "GET", "/snapshots/missing")

    assert artifact.status_code == 200
    assert dangling_artifact.status_code == 404
    assert dangling_artifact.json()["detail"]["error"]["type"] == "EventNotFoundError"
    assert invalid_base64.status_code == 400
    assert invalid_base64.json()["detail"]["error"]["type"] == "InvalidPayloadError"
    assert artifact_data.content == b"hello"
    assert artifact_data.headers["content-type"].startswith("text/plain")
    assert artifact_metadata.json()["media_type"] == "text/plain"
    assert artifact_metadata.json()["size"] == 5
    assert missing_artifact.status_code == 404
    assert missing_artifact_metadata.status_code == 404
    assert snapshot.json()["name"] == "latest"
    assert dangling_snapshot.status_code == 404
    assert dangling_snapshot.json()["detail"]["error"]["type"] == "EventNotFoundError"
    assert loaded.json()["value"] == {"status": "ok"}
    assert missing_snapshot.status_code == 404


def test_fastapi_mounts_custom_channel_endpoint(tmp_path):
    @endpoint.get("/channels/{name}/count")
    def count_events(store: LocalStore, name: str):
        return {"count": len(store.query_events(name))}

    @endpoint.put("/channels/{name}/state")
    def put_state(store: LocalStore, name: str, body=None):
        return {"name": name, "body": body}

    @endpoint.patch("/channels/{name}/state")
    def patch_state(store: LocalStore, name: str, body=None):
        return {"name": name, "patch": body}

    @endpoint.delete("/channels/{name}/state")
    def delete_state(store: LocalStore, name: str):
        return {"name": name, "deleted": True}

    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    store.append_event({"channel": "events", "payload": {"n": 1}})
    app = create_app(
        store,
        custom_endpoints=[count_events, put_state, patch_state, delete_state],
    )

    response = request(app, "GET", "/channels/events/count")
    put_response = request(
        app,
        "PUT",
        "/channels/events/state",
        json={"mode": "replace"},
    )
    patch_response = request(
        app,
        "PATCH",
        "/channels/events/state",
        json={"mode": "merge"},
    )
    delete_response = request(app, "DELETE", "/channels/events/state")

    assert response.status_code == 200
    assert response.json() == {"count": 1}
    assert put_response.status_code == 200
    assert put_response.json() == {"name": "events", "body": {"mode": "replace"}}
    assert patch_response.status_code == 200
    assert patch_response.json() == {"name": "events", "patch": {"mode": "merge"}}
    assert delete_response.status_code == 200
    assert delete_response.json() == {"name": "events", "deleted": True}


def test_fastapi_openapi_includes_portable_and_custom_routes(tmp_path):
    @endpoint.get("/channels/{name}/count")
    def count_events(store: LocalStore, name: str):
        return {"count": len(store.query_events(name))}

    @endpoint.put("/channels/{name}/state")
    def put_state(store: LocalStore, name: str, body=None):
        return {"name": name, "body": body}

    app = create_app(
        LocalStore(tmp_path / "store"),
        custom_endpoints=[count_events, put_state],
    )
    schema = app.openapi()

    for path in (
        "/health",
        "/channels",
        "/channels/{name}",
        "/events",
        "/subscriptions",
        "/subscriptions/{subscription_id}",
        "/subscriptions/{subscription_id}/pull",
        "/artifacts",
        "/artifacts/{artifact_id}",
        "/artifacts/{artifact_id}/metadata",
        "/events/{event_id}",
        "/snapshots/{name}",
        "/channels/{name}/count",
        "/channels/{name}/state",
    ):
        assert path in schema["paths"]
    assert "put" in schema["paths"]["/channels/{name}/state"]
    assert schema["paths"]["/channels/{name}/count"]["get"]["summary"] == "Count Events"
