import asyncio
import json
from types import MappingProxyType

import httpx
import pytest

from lllm.network import (
    AsyncRemoteStore,
    InvalidPayloadError,
    LocalStore,
    RemoteStore,
    RemoteStoreError,
    Snapshot,
)
from lllm.network.server import create_app


def test_async_client_calls_fastapi_server(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.ASGITransport(app=app),
        )
        channel = await client.create_channel({"name": "events"})
        event = await client.append_event({"channel": "events", "payload": {"n": 1}})
        analysis = await client.append_event(
            {"channel": "events", "kind": "analysis", "payload": {"n": 2}}
        )
        events = await client.query_events("events")
        loaded_event = await client.get_event(event.id)
        sub = await client.create_subscription(
            "events",
            subscription_id="worker-events",
        )
        pulled = await client.pull_subscription(sub.id)
        loaded_sub = await client.get_subscription(sub.id)
        reused_sub = await client.create_subscription(
            "events",
            subscription_id="worker-events",
        )
        filtered_sub = await client.create_subscription(
            "events",
            filters={"kind": "analysis"},
        )
        filtered = await client.pull_subscription(filtered_sub.id)
        artifact = await client.write_artifact(
            "hello",
            channel="events",
            metadata={"role": "raw"},
            event_ids=(event.id,),
        )
        loaded_artifact = await client.get_artifact(artifact.id)
        artifact_data = await client.read_artifact(artifact.id)
        binary_artifact = await client.write_artifact(
            b"\xff\x00binary",
            channel="events",
        )
        binary_artifact_data = await client.read_artifact(binary_artifact.id)
        snapshot = await client.put_snapshot(
            "latest",
            {"n": 1},
            channel="events",
            source_event_id=event.id,
            metadata={"role": "latest"},
        )
        loaded_snapshot = await client.get_snapshot("latest")
        model_snapshot = await client.put_snapshot(
            "model-latest",
            Snapshot(
                name="model-latest",
                channel="events",
                value={"n": 2},
                source_event_id=event.id,
                metadata={"role": "model"},
            ),
        )
        channels = await client.list_channels()
        loaded_channel = await client.get_channel("events")
        return {
            "channel": channel,
            "event": event,
            "analysis": analysis,
            "events": events,
            "loaded_event": loaded_event,
            "pulled": pulled,
            "loaded_sub": loaded_sub,
            "reused_sub": reused_sub,
            "filtered": filtered,
            "artifact": artifact,
            "loaded_artifact": loaded_artifact,
            "artifact_data": artifact_data,
            "binary_artifact_data": binary_artifact_data,
            "snapshot": snapshot,
            "loaded_snapshot": loaded_snapshot,
            "model_snapshot": model_snapshot,
            "channels": channels,
            "loaded_channel": loaded_channel,
        }

    result = asyncio.run(run())

    assert result["channel"].name == "events"
    assert result["event"].payload == {"n": 1}
    assert result["event"].cursor == 1
    assert result["analysis"].cursor == 2
    assert result["events"][0].id == result["event"].id
    assert result["loaded_event"].id == result["event"].id
    assert result["loaded_event"].payload == {"n": 1}
    assert result["pulled"][0].id == result["event"].id
    assert result["loaded_sub"].cursor == result["pulled"][-1].cursor
    assert result["reused_sub"].id == "worker-events"
    assert result["reused_sub"].cursor == result["loaded_sub"].cursor
    assert [event.id for event in result["filtered"]] == [result["analysis"].id]
    assert result["artifact"].metadata == {"role": "raw"}
    assert result["artifact"].event_ids == (result["event"].id,)
    assert result["loaded_artifact"].id == result["artifact"].id
    assert result["loaded_artifact"].metadata == {"role": "raw"}
    assert result["artifact_data"] == b"hello"
    assert result["binary_artifact_data"] == b"\xff\x00binary"
    assert result["snapshot"].name == "latest"
    assert result["snapshot"].value == {"n": 1}
    assert result["snapshot"].source_event_id == result["event"].id
    assert result["snapshot"].metadata == {"role": "latest"}
    assert result["loaded_snapshot"].value == {"n": 1}
    assert result["model_snapshot"].value == {"n": 2}
    assert result["model_snapshot"].metadata == {"role": "model"}
    assert result["channels"][0].name == "events"
    assert result["loaded_channel"].name == "events"


def test_async_client_preserves_fastapi_missing_resource_errors(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.ASGITransport(app=app),
        )
        cases = [
            (lambda: client.get_channel("missing"), "ChannelNotFoundError"),
            (lambda: client.query_events("missing"), "ChannelNotFoundError"),
            (lambda: client.get_event("missing"), "EventNotFoundError"),
            (
                lambda: client.pull_subscription("missing"),
                "SubscriptionNotFoundError",
            ),
            (
                lambda: client.get_subscription("missing"),
                "SubscriptionNotFoundError",
            ),
            (lambda: client.get_artifact("missing"), "ArtifactNotFoundError"),
            (lambda: client.read_artifact("missing"), "ArtifactNotFoundError"),
            (lambda: client.get_snapshot("missing"), "SnapshotNotFoundError"),
        ]

        for call, error_type in cases:
            with pytest.raises(RemoteStoreError) as exc_info:
                await call()
            assert exc_info.value.status_code == 404
            assert exc_info.value.error_type == error_type
            assert exc_info.value.message is not None
            assert "missing" in exc_info.value.message

    asyncio.run(run())


@pytest.mark.parametrize("client_cls", [RemoteStore, AsyncRemoteStore])
@pytest.mark.parametrize(
    "base_url",
    [
        None,
        123,
        "",
        "   ",
        "testserver",
        "/api",
        "ftp://testserver",
        " http://testserver ",
        "http://test server",
    ],
)
def test_http_clients_reject_malformed_base_urls(client_cls, base_url):
    with pytest.raises(ValueError, match="base_url"):
        client_cls(base_url)  # type: ignore[arg-type]


def test_sync_client_uses_portable_api_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/channels"
        return httpx.Response(
            200,
            json={"name": "events", "schema": None, "form": "log", "description": "", "metadata": {}},
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    assert client.create_channel({"name": "events"}).name == "events"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, text="{bad"), "not valid JSON"),
        (httpx.Response(200, json={"name": 123}), "expected schema"),
        (httpx.Response(200, json={"metadata": {}}), "expected schema"),
    ],
)
def test_sync_client_reports_invalid_success_responses(response, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/channels/events"
        return response

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    with pytest.raises(RemoteStoreError) as exc_info:
        client.get_channel("events")

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_type == "InvalidResponse"
    assert expected in exc_info.value.message


def test_async_client_reports_invalid_success_responses():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/channels"
        return httpx.Response(200, json={"name": "events"})

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        await client.list_channels()

    with pytest.raises(RemoteStoreError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_type == "InvalidResponse"
    assert "expected schema" in exc_info.value.message


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("error_type", {"error_type": b"InvalidResponse"}),
        ("message", {"message": b"bad"}),
    ],
)
def test_client_error_rejects_bytes_for_text_fields(field_name, kwargs):
    with pytest.raises(TypeError, match=field_name):
        RemoteStoreError(500, **kwargs)


@pytest.mark.parametrize(
    "error_type",
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
def test_client_error_rejects_malformed_error_type_tokens(error_type):
    with pytest.raises(ValueError, match="error_type"):
        RemoteStoreError(500, error_type=error_type)


def test_sync_client_ignores_non_string_error_payload_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/channels/events"
        return httpx.Response(
            500,
            json={"detail": {"error": {"type": 123, "message": ["bad"]}}},
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    with pytest.raises(RemoteStoreError) as exc_info:
        client.get_channel("events")

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type is None
    assert exc_info.value.message is None


def test_sync_client_ignores_malformed_error_type_payload_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/channels/events"
        return httpx.Response(
            500,
            json={
                "detail": {
                    "error": {
                        "type": "bad type",
                        "message": "still readable",
                    }
                }
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    with pytest.raises(RemoteStoreError) as exc_info:
        client.get_channel("events")

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type is None
    assert exc_info.value.message == "still readable"


@pytest.mark.parametrize("bad_name", ["bad/name", "bad name", "bad%2Fname"])
def test_sync_client_rejects_path_control_lookup_names_without_request(bad_name):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))
    calls = (
        lambda: client.get_channel(bad_name),
        lambda: client.query_events(bad_name),
        lambda: client.get_event(bad_name),
        lambda: client.pull_subscription(bad_name),
        lambda: client.get_subscription(bad_name),
        lambda: client.get_artifact(bad_name),
        lambda: client.read_artifact(bad_name),
        lambda: client.put_snapshot(bad_name, {}),
        lambda: client.get_snapshot(bad_name),
    )

    for call in calls:
        with pytest.raises(InvalidPayloadError):
            call()


@pytest.mark.parametrize("bad_name", ["bad/name", "bad name", "bad%2Fname"])
def test_sync_client_rejects_path_control_body_ids_without_request(bad_name):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))
    calls = (
        lambda: client.create_channel({"name": bad_name}),
        lambda: client.append_event({"id": bad_name, "channel": "events"}),
        lambda: client.append_event({"channel": bad_name}),
        lambda: client.append_event(
            {"channel": "events", "parent_ids": [bad_name]}
        ),
        lambda: client.create_subscription(bad_name),
        lambda: client.create_subscription("events", subscription_id=bad_name),
        lambda: client.write_artifact("hello", channel=bad_name),
        lambda: client.write_artifact("hello", event_ids=(bad_name,)),
        lambda: client.put_snapshot("latest", {}, channel=bad_name),
        lambda: client.put_snapshot("latest", {}, source_event_id=bad_name),
        lambda: client.put_snapshot(
            {"name": "latest", "channel": bad_name, "value": {}}
        ),
    )

    for call in calls:
        with pytest.raises(InvalidPayloadError):
            call()


@pytest.mark.parametrize("bad_data", [None, 123, {"data": "hello"}])
def test_sync_client_rejects_malformed_artifact_data_without_request(bad_data):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    with pytest.raises(InvalidPayloadError, match="artifact data"):
        client.write_artifact(bad_data)  # type: ignore[arg-type]


def test_sync_client_rejects_malformed_payload_shapes_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))
    calls = (
        (lambda: client.create_channel(["bad"]), "channel"),
        (lambda: client.create_channel({"name": "events", "metadata": []}), "metadata"),
        (lambda: client.append_event(["bad"]), "event"),
        (lambda: client.append_event({"channel": "events", "metadata": []}), "metadata"),
        (
            lambda: client.append_event(
                {"channel": "events", "parent_ids": "event-1"}
            ),
            "parent_ids",
        ),
        (lambda: client.create_subscription("events", filters=[]), "filters"),
        (lambda: client.create_subscription("events", metadata=[]), "metadata"),
        (lambda: client.write_artifact("hello", metadata=[]), "metadata"),
        (lambda: client.write_artifact("hello", event_ids="event-1"), "event_ids"),
        (lambda: client.put_snapshot("latest", {}, metadata=[]), "metadata"),
        (
            lambda: client.put_snapshot(
                {"name": "latest", "value": {}, "metadata": []}
            ),
            "metadata",
        ),
    )

    for call, message in calls:
        with pytest.raises(InvalidPayloadError, match=message):
            call()


@pytest.mark.parametrize(
    "bad_kind",
    (
        "",
        "   ",
        ".",
        "..",
        "bad kind",
        "bad/kind",
        "bad:kind",
        "bad\\kind",
        "bad%2Fkind",
    ),
)
def test_sync_client_rejects_malformed_event_kind_without_request(bad_kind):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))
    calls = (
        lambda: client.append_event({"channel": "events", "kind": bad_kind}),
        lambda: client.query_events("events", kind=bad_kind),
    )

    for call in calls:
        with pytest.raises(InvalidPayloadError, match="event.kind"):
            call()


@pytest.mark.parametrize(
    "bad_id",
    (
        "",
        "   ",
        ".",
        "..",
        "bad id",
        "bad/id",
        "bad:id",
        "bad\\id",
        "bad%2Fid",
    ),
)
def test_sync_client_rejects_malformed_coordination_ids_without_request(bad_id):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))
    calls = (
        (
            lambda: client.append_event(
                {"channel": "events", "correlation_id": bad_id}
            ),
            "event.correlation_id",
        ),
        (
            lambda: client.create_subscription("events", consumer=bad_id),
            "subscription.consumer",
        ),
    )

    for call, field_name in calls:
        with pytest.raises(InvalidPayloadError, match=field_name):
            call()


@pytest.mark.parametrize("bad_name", ["bad/name", "bad name", "bad%2Fname"])
def test_async_client_rejects_path_control_lookup_names_without_request(bad_name):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        calls = (
            lambda: client.get_channel(bad_name),
            lambda: client.query_events(bad_name),
            lambda: client.get_event(bad_name),
            lambda: client.pull_subscription(bad_name),
            lambda: client.get_subscription(bad_name),
            lambda: client.get_artifact(bad_name),
            lambda: client.read_artifact(bad_name),
            lambda: client.put_snapshot(bad_name, {}),
            lambda: client.get_snapshot(bad_name),
        )

        for call in calls:
            with pytest.raises(InvalidPayloadError):
                await call()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_kind",
    (
        "",
        "   ",
        ".",
        "..",
        "bad kind",
        "bad/kind",
        "bad:kind",
        "bad\\kind",
        "bad%2Fkind",
    ),
)
def test_async_client_rejects_malformed_event_kind_without_request(bad_kind):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        calls = (
            lambda: client.append_event({"channel": "events", "kind": bad_kind}),
            lambda: client.query_events("events", kind=bad_kind),
        )

        for call in calls:
            with pytest.raises(InvalidPayloadError, match="event.kind"):
                await call()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_id",
    (
        "",
        "   ",
        ".",
        "..",
        "bad id",
        "bad/id",
        "bad:id",
        "bad\\id",
        "bad%2Fid",
    ),
)
def test_async_client_rejects_malformed_coordination_ids_without_request(bad_id):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        calls = (
            (
                lambda: client.append_event(
                    {"channel": "events", "correlation_id": bad_id}
                ),
                "event.correlation_id",
            ),
            (
                lambda: client.create_subscription("events", consumer=bad_id),
                "subscription.consumer",
            ),
        )

        for call, field_name in calls:
            with pytest.raises(InvalidPayloadError, match=field_name):
                await call()

    asyncio.run(run())


@pytest.mark.parametrize("bad_name", ["bad/name", "bad name", "bad%2Fname"])
def test_async_client_rejects_path_control_body_ids_without_request(bad_name):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        calls = (
            lambda: client.create_channel({"name": bad_name}),
            lambda: client.append_event({"id": bad_name, "channel": "events"}),
            lambda: client.append_event({"channel": bad_name}),
            lambda: client.append_event(
                {"channel": "events", "parent_ids": [bad_name]}
            ),
            lambda: client.create_subscription(bad_name),
            lambda: client.create_subscription("events", subscription_id=bad_name),
            lambda: client.write_artifact("hello", channel=bad_name),
            lambda: client.write_artifact("hello", event_ids=(bad_name,)),
            lambda: client.put_snapshot("latest", {}, channel=bad_name),
            lambda: client.put_snapshot("latest", {}, source_event_id=bad_name),
            lambda: client.put_snapshot(
                {"name": "latest", "channel": bad_name, "value": {}}
            ),
        )

        for call in calls:
            with pytest.raises(InvalidPayloadError):
                await call()

    asyncio.run(run())


@pytest.mark.parametrize("bad_data", [None, 123, {"data": "hello"}])
def test_async_client_rejects_malformed_artifact_data_without_request(bad_data):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        with pytest.raises(InvalidPayloadError, match="artifact data"):
            await client.write_artifact(bad_data)  # type: ignore[arg-type]

    asyncio.run(run())


def test_async_client_rejects_malformed_payload_shapes_without_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.MockTransport(handler),
        )
        calls = (
            (lambda: client.create_channel(["bad"]), "channel"),
            (
                lambda: client.create_channel(
                    {"name": "events", "metadata": []}
                ),
                "metadata",
            ),
            (lambda: client.append_event(["bad"]), "event"),
            (
                lambda: client.append_event(
                    {"channel": "events", "metadata": []}
                ),
                "metadata",
            ),
            (
                lambda: client.append_event(
                    {"channel": "events", "parent_ids": "event-1"}
                ),
                "parent_ids",
            ),
            (lambda: client.create_subscription("events", filters=[]), "filters"),
            (lambda: client.create_subscription("events", metadata=[]), "metadata"),
            (lambda: client.write_artifact("hello", metadata=[]), "metadata"),
            (
                lambda: client.write_artifact("hello", event_ids="event-1"),
                "event_ids",
            ),
            (lambda: client.put_snapshot("latest", {}, metadata=[]), "metadata"),
            (
                lambda: client.put_snapshot(
                    {"name": "latest", "value": {}, "metadata": []},
                ),
                "metadata",
            ),
        )

        for call, message in calls:
            with pytest.raises(InvalidPayloadError, match=message):
                await call()

    asyncio.run(run())


def test_sync_client_sends_artifact_metadata_and_event_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/artifacts"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {
            "data": "hello",
            "encoding": "text",
            "channel": "events",
            "media_type": "text/plain",
            "metadata": {"role": "raw"},
            "event_ids": ["event-1"],
        }
        return httpx.Response(
            200,
            json={
                "id": "artifact-1",
                "channel": "events",
                "path": "artifacts/artifact-1",
                "media_type": "text/plain",
                "size": 5,
                "metadata": {"role": "raw"},
                "event_ids": ["event-1"],
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    artifact = client.write_artifact(
        "hello",
        channel="events",
        media_type="text/plain",
        metadata={"role": "raw"},
        event_ids=("event-1",),
    )

    assert artifact.metadata == {"role": "raw"}
    assert artifact.event_ids == ("event-1",)


def test_sync_client_sends_subscription_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/subscriptions"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {
            "id": "worker-events",
            "channel": "events",
            "consumer": "worker",
            "batch_size": 10,
            "filters": {"kind": "raw"},
            "metadata": {"role": "processor"},
        }
        return httpx.Response(
            200,
            json={
                "id": "worker-events",
                "channel": "events",
                "cursor": 0,
                "consumer": "worker",
                "batch_size": 10,
                "filters": {"kind": "raw"},
                "metadata": {"role": "processor"},
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    sub = client.create_subscription(
        "events",
        subscription_id="worker-events",
        consumer="worker",
        batch_size=10,
        filters={"kind": "raw"},
        metadata={"role": "processor"},
    )

    assert sub.id == "worker-events"
    assert sub.filters == {"kind": "raw"}


def test_sync_client_isolates_mutable_request_inputs():
    class CapturingClient(RemoteStore):
        def __init__(self):
            super().__init__("http://testserver")
            self.payloads = []

        def _request(self, method, path, **kwargs):
            self.payloads.append(kwargs["json"])
            if path == "/channels":
                return httpx.Response(
                    200,
                    json={
                        "name": "events",
                        "schema": None,
                        "form": "log",
                        "description": "",
                        "metadata": {},
                    },
                )
            if path == "/events":
                return httpx.Response(
                    200,
                    json={
                        "id": "event-1",
                        "cursor": 1,
                        "channel": "events",
                        "timestamp": 1.0,
                        "source": None,
                        "kind": "raw",
                        "payload": {"items": ["event"]},
                        "schema": None,
                        "metadata": {"labels": ["event"]},
                        "correlation_id": None,
                        "parent_ids": [],
                    },
                )
            if path == "/subscriptions":
                return httpx.Response(
                    200,
                    json={
                        "id": "worker",
                        "channel": "events",
                        "cursor": 0,
                        "consumer": None,
                        "batch_size": 100,
                        "filters": {"kind": {"labels": ["raw"]}},
                        "metadata": {"labels": ["sub"]},
                    },
                )
            if path == "/artifacts":
                return httpx.Response(
                    200,
                    json={
                        "id": "artifact-1",
                        "channel": "events",
                        "path": "artifacts/artifact-1",
                        "media_type": "text/plain",
                        "size": 5,
                        "metadata": {"labels": ["artifact"]},
                        "event_ids": [],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "name": "latest",
                    "channel": "events",
                    "timestamp": 1.0,
                    "value": {"items": ["snapshot"]},
                    "metadata": {"labels": ["snapshot"]},
                },
            )

    client = CapturingClient()
    channel = {"name": "events", "metadata": {"labels": ["channel"]}}
    event = {
        "channel": "events",
        "kind": "raw",
        "payload": {"items": ["event"]},
        "metadata": {"labels": ["event"]},
    }
    filters = {"kind": {"labels": ["raw"]}}
    sub_metadata = {"labels": ["sub"]}
    artifact_metadata = {"labels": ["artifact"]}
    snapshot_value = {"items": ["snapshot"]}
    snapshot_metadata = {"labels": ["snapshot"]}

    client.create_channel(channel)
    client.append_event(event)
    client.create_subscription(
        "events",
        subscription_id="worker",
        filters=filters,
        metadata=sub_metadata,
    )
    client.write_artifact(
        "hello",
        channel="events",
        media_type="text/plain",
        metadata=artifact_metadata,
    )
    client.put_snapshot(
        "latest",
        snapshot_value,
        channel="events",
        metadata=snapshot_metadata,
    )

    channel["metadata"]["labels"].append("changed")
    event["payload"]["items"].append("changed")
    event["metadata"]["labels"].append("changed")
    filters["kind"]["labels"].append("changed")
    sub_metadata["labels"].append("changed")
    artifact_metadata["labels"].append("changed")
    snapshot_value["items"].append("changed")
    snapshot_metadata["labels"].append("changed")

    assert client.payloads[0]["metadata"] == {"labels": ["channel"]}
    assert client.payloads[1]["payload"] == {"items": ["event"]}
    assert client.payloads[1]["metadata"] == {"labels": ["event"]}
    assert client.payloads[2]["filters"] == {"kind": {"labels": ["raw"]}}
    assert client.payloads[2]["metadata"] == {"labels": ["sub"]}
    assert client.payloads[3]["metadata"] == {"labels": ["artifact"]}
    assert client.payloads[4]["value"] == {"items": ["snapshot"]}
    assert client.payloads[4]["metadata"] == {"labels": ["snapshot"]}


def test_sync_client_accepts_nested_read_only_mapping_request_inputs():
    class CapturingClient(RemoteStore):
        def __init__(self):
            super().__init__("http://testserver")
            self.payloads = []

        def _request(self, method, path, **kwargs):
            self.payloads.append(kwargs["json"])
            if path == "/channels":
                return httpx.Response(
                    200,
                    json={
                        "name": "events",
                        "schema": None,
                        "form": "log",
                        "description": "",
                        "metadata": {},
                    },
                )
            if path == "/events":
                return httpx.Response(
                    200,
                    json={
                        "id": "event-1",
                        "cursor": 1,
                        "channel": "events",
                        "timestamp": 1.0,
                        "source": None,
                        "kind": "raw",
                        "payload": {"nested": {"item": "event"}},
                        "schema": None,
                        "metadata": {"nested": {"label": "event"}},
                        "correlation_id": None,
                        "parent_ids": [],
                    },
                )
            if path == "/subscriptions":
                return httpx.Response(
                    200,
                    json={
                        "id": "worker",
                        "channel": "events",
                        "cursor": 0,
                        "consumer": None,
                        "batch_size": 100,
                        "filters": {"nested": {"kind": "raw"}},
                        "metadata": {"nested": {"label": "sub"}},
                    },
                )
            if path == "/artifacts":
                return httpx.Response(
                    200,
                    json={
                        "id": "artifact-1",
                        "channel": "events",
                        "path": "artifacts/artifact-1",
                        "media_type": "text/plain",
                        "size": 5,
                        "metadata": {"nested": {"label": "artifact"}},
                        "event_ids": [],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "name": "latest",
                    "channel": "events",
                    "timestamp": 1.0,
                    "value": {"nested": {"item": "snapshot"}},
                    "metadata": {"nested": {"label": "snapshot"}},
                },
            )

    client = CapturingClient()
    channel_inner = {"label": "channel"}
    payload_inner = {"item": "event"}
    event_inner = {"label": "event"}
    filter_inner = {"kind": "raw"}
    sub_inner = {"label": "sub"}
    artifact_inner = {"label": "artifact"}
    snapshot_value_inner = {"item": "snapshot"}
    snapshot_inner = {"label": "snapshot"}

    client.create_channel(
        {"name": "events", "metadata": {"nested": MappingProxyType(channel_inner)}}
    )
    client.append_event(
        {
            "channel": "events",
            "kind": "raw",
            "payload": {"nested": MappingProxyType(payload_inner)},
            "metadata": {"nested": MappingProxyType(event_inner)},
        }
    )
    client.create_subscription(
        "events",
        subscription_id="worker",
        filters={"nested": MappingProxyType(filter_inner)},
        metadata={"nested": MappingProxyType(sub_inner)},
    )
    client.write_artifact(
        "hello",
        channel="events",
        metadata={"nested": MappingProxyType(artifact_inner)},
    )
    client.put_snapshot(
        "latest",
        {"nested": MappingProxyType(snapshot_value_inner)},
        channel="events",
        metadata={"nested": MappingProxyType(snapshot_inner)},
    )

    channel_inner["label"] = "changed"
    payload_inner["item"] = "changed"
    event_inner["label"] = "changed"
    filter_inner["kind"] = "changed"
    sub_inner["label"] = "changed"
    artifact_inner["label"] = "changed"
    snapshot_value_inner["item"] = "changed"
    snapshot_inner["label"] = "changed"

    assert client.payloads[0]["metadata"] == {"nested": {"label": "channel"}}
    assert client.payloads[1]["payload"] == {"nested": {"item": "event"}}
    assert client.payloads[1]["metadata"] == {"nested": {"label": "event"}}
    assert client.payloads[2]["filters"] == {"nested": {"kind": "raw"}}
    assert client.payloads[2]["metadata"] == {"nested": {"label": "sub"}}
    assert client.payloads[3]["metadata"] == {"nested": {"label": "artifact"}}
    assert client.payloads[4]["value"] == {"nested": {"item": "snapshot"}}
    assert client.payloads[4]["metadata"] == {"nested": {"label": "snapshot"}}


def test_sync_client_gets_artifact_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/artifacts/artifact-1/metadata"
        return httpx.Response(
            200,
            json={
                "id": "artifact-1",
                "channel": "events",
                "path": "artifacts/artifact-1",
                "media_type": "text/plain",
                "size": 5,
                "sha256": "abc123",
                "metadata": {"role": "raw"},
                "event_ids": ["event-1"],
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    artifact = client.get_artifact("artifact-1")

    assert artifact.media_type == "text/plain"
    assert artifact.metadata == {"role": "raw"}
    assert artifact.event_ids == ("event-1",)


def test_sync_client_gets_event_by_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/events/event-1"
        return httpx.Response(
            200,
            json={
                "id": "event-1",
                "cursor": 7,
                "channel": "events",
                "timestamp": 1.0,
                "source": "test",
                "kind": "raw",
                "payload": {"n": 1},
                "schema": None,
                "metadata": {},
                "correlation_id": None,
                "parent_ids": [],
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    event = client.get_event("event-1")

    assert event.id == "event-1"
    assert event.cursor == 7
    assert event.payload == {"n": 1}


def test_sync_client_sends_snapshot_metadata_and_source_event():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/snapshots/latest"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {
            "value": {"status": "ok"},
            "channel": "events",
            "schema": "demo.schemas:State",
            "source_event_id": "event-1",
            "metadata": {"role": "latest"},
        }
        return httpx.Response(
            200,
            json={
                "name": "latest",
                "channel": "events",
                "timestamp": 1.0,
                "value": {"status": "ok"},
                "schema": "demo.schemas:State",
                "source_event_id": "event-1",
                "metadata": {"role": "latest"},
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    snapshot = client.put_snapshot(
        "latest",
        {"status": "ok"},
        channel="events",
        schema="demo.schemas:State",
        source_event_id="event-1",
        metadata={"role": "latest"},
    )

    assert snapshot.value == {"status": "ok"}
    assert snapshot.schema == "demo.schemas:State"
    assert snapshot.source_event_id == "event-1"
    assert snapshot.metadata == {"role": "latest"}


def test_sync_client_sends_snapshot_model_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/snapshots/latest"
        body = json.loads(request.content.decode("utf-8"))
        assert isinstance(body.pop("timestamp"), float)
        assert body == {
            "channel": "events",
            "value": {"status": "ok"},
            "source_event_id": "event-1",
            "metadata": {"role": "latest"},
        }
        return httpx.Response(
            200,
            json={
                "name": "latest",
                "channel": "events",
                "timestamp": 1.0,
                "value": {"status": "ok"},
                "source_event_id": "event-1",
                "metadata": {"role": "latest"},
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    snapshot = client.put_snapshot(
        "latest",
        Snapshot(
            name="latest",
            channel="events",
            value={"status": "ok"},
            source_event_id="event-1",
            metadata={"role": "latest"},
        ),
    )

    assert snapshot.value == {"status": "ok"}
    assert snapshot.source_event_id == "event-1"


def test_sync_client_snapshot_model_allows_field_overrides():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["channel"] == "override-channel"
        assert body["schema"] == "demo.schemas:Override"
        assert body["source_event_id"] == "override-event"
        assert body["metadata"] == {"role": "override"}
        assert body["value"] == {"status": "ok"}
        assert "name" not in body
        return httpx.Response(
            200,
            json={
                "name": "latest",
                "channel": body["channel"],
                "timestamp": body["timestamp"],
                "value": body["value"],
                "schema": body["schema"],
                "source_event_id": body["source_event_id"],
                "metadata": body["metadata"],
            },
        )

    client = RemoteStore("http://testserver", transport=httpx.MockTransport(handler))

    snapshot = client.put_snapshot(
        "latest",
        Snapshot(
            name="latest",
            channel="events",
            value={"status": "ok"},
            source_event_id="event-1",
            metadata={"role": "latest"},
        ),
        channel="override-channel",
        schema="demo.schemas:Override",
        source_event_id="override-event",
        metadata={"role": "override"},
    )

    assert snapshot.channel == "override-channel"
    assert snapshot.schema == "demo.schemas:Override"
    assert snapshot.source_event_id == "override-event"
    assert snapshot.metadata == {"role": "override"}


def test_async_client_maps_server_errors(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.ASGITransport(app=app),
        )
        await client.get_channel("missing")

    with pytest.raises(RemoteStoreError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.status_code == 404
    assert exc_info.value.error_type == "ChannelNotFoundError"
    assert exc_info.value.message == "Channel not found: missing"


def test_async_client_maps_subscription_conflicts(tmp_path):
    app = create_app(LocalStore(tmp_path / "store"))

    async def run():
        client = AsyncRemoteStore(
            "http://testserver",
            transport=httpx.ASGITransport(app=app),
        )
        await client.create_channel({"name": "events"})
        await client.create_channel({"name": "other-events"})
        await client.create_subscription("events", subscription_id="worker-events")
        await client.create_subscription(
            "other-events",
            subscription_id="worker-events",
        )

    with pytest.raises(RemoteStoreError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_type == "SubscriptionExistsError"
