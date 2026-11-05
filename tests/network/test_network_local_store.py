import sqlite3
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from lllm.network import (
    Artifact,
    ArtifactNotFoundError,
    Channel,
    ChannelExistsError,
    ChannelNotFoundError,
    Event,
    EventNotFoundError,
    InvalidPayloadError,
    LocalStore,
    Snapshot,
    SnapshotNotFoundError,
    Subscription,
    SubscriptionExistsError,
    SubscriptionNotFoundError,
)


def test_store_initializes_deterministic_layout(tmp_path):
    store = LocalStore(tmp_path / "store")

    assert store.db_path == (tmp_path / "store" / "network.sqlite").resolve()
    assert store.artifacts_dir == (tmp_path / "store" / "artifacts").resolve()
    assert store.db_path.exists()
    assert store.artifacts_dir.is_dir()


def test_store_rejects_blank_or_non_path_roots(tmp_path):
    padded_root = tmp_path / "padded"
    for root in ("   ", 123, f"  {padded_root}  "):
        with pytest.raises(ValueError, match="store root"):
            LocalStore(root)  # type: ignore[arg-type]

    assert not padded_root.exists()


def test_store_reopens_persisted_resources(tmp_path):
    root = tmp_path / "store"
    store = LocalStore(root)
    channel = store.create_channel({"name": "events", "schema": "demo.schemas:Event"})
    event = store.append_event({"channel": "events", "payload": {"status": "ok"}})
    subscription = store.create_subscription(
        "events",
        subscription_id="worker",
        consumer="analyzer",
    )
    artifact = store.write_artifact(
        b"payload",
        channel="events",
        media_type="text/plain",
        event_ids=(event.id,),
    )
    snapshot = store.put_snapshot(
        {"name": "latest", "channel": "events", "value": {"status": "ok"}}
    )

    reopened = LocalStore(root)

    assert reopened.list_channels() == (channel,)
    assert reopened.get_event(event.id) == event
    assert reopened.get_subscription(subscription.id) == subscription
    assert reopened.get_artifact(artifact.id) == artifact
    assert reopened.read_artifact(artifact.id) == b"payload"
    assert reopened.get_snapshot(snapshot.name) == snapshot


def test_channel_create_list_get_and_errors(tmp_path):
    store = LocalStore(tmp_path / "store")
    channel = store.create_channel(
        Channel(name="events", schema="demo.schemas:Event", form="log")
    )

    assert channel.name == "events"
    assert store.get_channel("events").schema == "demo.schemas:Event"
    assert store.list_channels() == (channel,)
    with pytest.raises(ChannelExistsError):
        store.create_channel({"name": "events"})
    with pytest.raises(ChannelNotFoundError):
        store.get_channel("missing")


@pytest.mark.parametrize("metadata", [[], [("owner", "tests")], "bad", 123])
def test_local_store_create_channel_rejects_non_object_metadata(tmp_path, metadata):
    store = LocalStore(tmp_path / "store")

    with pytest.raises(InvalidPayloadError, match="channel.metadata"):
        store.create_channel({"name": "events", "metadata": metadata})

    with pytest.raises(ChannelNotFoundError):
        store.get_channel("events")


def test_core_models_isolate_mutable_constructor_inputs():
    channel_metadata = {"labels": ["raw"]}
    event_payload = {"items": ["event"]}
    event_metadata = {"labels": ["event"]}
    artifact_metadata = {"labels": ["artifact"]}
    snapshot_value = {"items": ["snapshot"]}
    snapshot_metadata = {"labels": ["snapshot"]}
    subscription_filters = {"kind": ["raw"]}
    subscription_metadata = {"labels": ["subscription"]}

    channel = Channel(name="events", metadata=channel_metadata)
    event = Event(
        channel="events",
        payload=event_payload,
        metadata=event_metadata,
    )
    artifact = Artifact(
        id="artifact",
        path="artifact.bin",
        size=1,
        metadata=artifact_metadata,
    )
    snapshot = Snapshot(
        name="latest",
        value=snapshot_value,
        metadata=snapshot_metadata,
    )
    subscription = Subscription(
        id="worker",
        channel="events",
        filters=subscription_filters,
        metadata=subscription_metadata,
    )

    channel_metadata["labels"].append("changed")
    event_payload["items"].append("changed")
    event_metadata["labels"].append("changed")
    artifact_metadata["labels"].append("changed")
    snapshot_value["items"].append("changed")
    snapshot_metadata["labels"].append("changed")
    subscription_filters["kind"].append("changed")
    subscription_metadata["labels"].append("changed")

    assert channel.metadata == {"labels": ["raw"]}
    assert event.payload == {"items": ["event"]}
    assert event.metadata == {"labels": ["event"]}
    assert artifact.metadata == {"labels": ["artifact"]}
    assert snapshot.value == {"items": ["snapshot"]}
    assert snapshot.metadata == {"labels": ["snapshot"]}
    assert subscription.filters == {"kind": ["raw"]}
    assert subscription.metadata == {"labels": ["subscription"]}


def test_core_models_accept_nested_read_only_mapping_inputs():
    channel_inner = {"label": "raw"}
    payload_inner = {"item": "event"}
    event_inner = {"label": "event"}
    artifact_inner = {"label": "artifact"}
    snapshot_value_inner = {"item": "snapshot"}
    snapshot_inner = {"label": "snapshot"}
    filter_inner = {"kind": "raw"}
    subscription_inner = {"label": "subscription"}

    channel = Channel(name="events", metadata={"nested": MappingProxyType(channel_inner)})
    event = Event(
        channel="events",
        payload={"nested": MappingProxyType(payload_inner)},
        metadata={"nested": MappingProxyType(event_inner)},
    )
    artifact = Artifact(
        id="artifact",
        path="artifact.bin",
        size=1,
        metadata={"nested": MappingProxyType(artifact_inner)},
    )
    snapshot = Snapshot(
        name="latest",
        value={"nested": MappingProxyType(snapshot_value_inner)},
        metadata={"nested": MappingProxyType(snapshot_inner)},
    )
    subscription = Subscription(
        id="worker",
        channel="events",
        filters={"nested": MappingProxyType(filter_inner)},
        metadata={"nested": MappingProxyType(subscription_inner)},
    )

    channel_inner["label"] = "changed"
    payload_inner["item"] = "changed"
    event_inner["label"] = "changed"
    artifact_inner["label"] = "changed"
    snapshot_value_inner["item"] = "changed"
    snapshot_inner["label"] = "changed"
    filter_inner["kind"] = "changed"
    subscription_inner["label"] = "changed"

    assert channel.metadata == {"nested": {"label": "raw"}}
    assert event.payload == {"nested": {"item": "event"}}
    assert event.metadata == {"nested": {"label": "event"}}
    assert artifact.metadata == {"nested": {"label": "artifact"}}
    assert snapshot.value == {"nested": {"item": "snapshot"}}
    assert snapshot.metadata == {"nested": {"label": "snapshot"}}
    assert subscription.filters == {"nested": {"kind": "raw"}}
    assert subscription.metadata == {"nested": {"label": "subscription"}}


def test_store_returns_isolated_mutable_write_inputs(tmp_path):
    store = LocalStore(tmp_path / "store")
    channel_input = Channel(name="events", metadata={"labels": ["raw"]})
    channel = store.create_channel(channel_input)
    channel_input.metadata["labels"].append("changed")
    assert channel.metadata == {"labels": ["raw"]}
    assert store.get_channel("events").metadata == {"labels": ["raw"]}

    event_input = Event(
        channel="events",
        payload={"items": ["one"]},
        metadata={"labels": ["event"]},
    )
    event = store.append_event(event_input)
    event_input.payload["items"].append("changed")
    event_input.metadata["labels"].append("changed")
    assert event.payload == {"items": ["one"]}
    assert event.metadata == {"labels": ["event"]}
    assert store.get_event(event.id).payload == {"items": ["one"]}

    event_payload = {
        "channel": "events",
        "payload": {"items": ["two"]},
        "metadata": {"labels": ["dict"]},
    }
    dict_event = store.append_event(event_payload)
    event_payload["payload"]["items"].append("changed")
    event_payload["metadata"]["labels"].append("changed")
    assert dict_event.payload == {"items": ["two"]}
    assert dict_event.metadata == {"labels": ["dict"]}

    filters = {"kind": "event"}
    subscription_metadata = {"labels": ["sub"]}
    subscription = store.create_subscription(
        "events",
        filters=filters,
        metadata=subscription_metadata,
    )
    filters["kind"] = "changed"
    subscription_metadata["labels"].append("changed")
    assert subscription.filters == {"kind": "event"}
    assert subscription.metadata == {"labels": ["sub"]}
    assert store.get_subscription(subscription.id).filters == {"kind": "event"}

    artifact_metadata = {"labels": ["artifact"]}
    artifact = store.write_artifact(b"payload", metadata=artifact_metadata)
    artifact_metadata["labels"].append("changed")
    assert artifact.metadata == {"labels": ["artifact"]}
    assert store.get_artifact(artifact.id).metadata == {"labels": ["artifact"]}

    snapshot_input = Snapshot(
        name="latest",
        channel="events",
        value={"items": ["snapshot"]},
        metadata={"labels": ["snapshot"]},
    )
    snapshot = store.put_snapshot(snapshot_input)
    snapshot_input.value["items"].append("changed")
    snapshot_input.metadata["labels"].append("changed")
    assert snapshot.value == {"items": ["snapshot"]}
    assert snapshot.metadata == {"labels": ["snapshot"]}
    assert store.get_snapshot("latest").value == {"items": ["snapshot"]}


def test_store_accepts_nested_read_only_mapping_write_inputs(tmp_path):
    store = LocalStore(tmp_path / "store")
    channel_inner = {"label": "raw"}
    channel = store.create_channel(
        {"name": "events", "metadata": {"nested": MappingProxyType(channel_inner)}}
    )
    payload_inner = {"item": "event"}
    event_inner = {"label": "event"}
    event = store.append_event(
        {
            "channel": "events",
            "payload": {"nested": MappingProxyType(payload_inner)},
            "metadata": {"nested": MappingProxyType(event_inner)},
        }
    )
    subscription_inner = {"label": "subscription"}
    subscription = store.create_subscription(
        "events",
        filters={"kind": "event"},
        metadata={"nested": MappingProxyType(subscription_inner)},
    )
    artifact_inner = {"label": "artifact"}
    artifact = store.write_artifact(
        b"payload",
        metadata={"nested": MappingProxyType(artifact_inner)},
    )
    snapshot_value_inner = {"item": "snapshot"}
    snapshot_inner = {"label": "snapshot"}
    snapshot = store.put_snapshot(
        {
            "name": "latest",
            "value": {"nested": MappingProxyType(snapshot_value_inner)},
            "metadata": {"nested": MappingProxyType(snapshot_inner)},
        }
    )

    channel_inner["label"] = "changed"
    payload_inner["item"] = "changed"
    event_inner["label"] = "changed"
    subscription_inner["label"] = "changed"
    artifact_inner["label"] = "changed"
    snapshot_value_inner["item"] = "changed"
    snapshot_inner["label"] = "changed"

    assert channel.metadata == {"nested": {"label": "raw"}}
    assert event.payload == {"nested": {"item": "event"}}
    assert event.metadata == {"nested": {"label": "event"}}
    assert subscription.filters == {"kind": "event"}
    assert subscription.metadata == {"nested": {"label": "subscription"}}
    assert artifact.metadata == {"nested": {"label": "artifact"}}
    assert snapshot.value == {"nested": {"item": "snapshot"}}
    assert snapshot.metadata == {"nested": {"label": "snapshot"}}
    assert store.get_channel("events").metadata == {"nested": {"label": "raw"}}
    assert store.get_event(event.id).payload == {"nested": {"item": "event"}}
    assert store.get_artifact(artifact.id).metadata == {
        "nested": {"label": "artifact"}
    }
    assert store.get_snapshot("latest").value == {"nested": {"item": "snapshot"}}


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        ".",
        "..",
        "bad/name",
        r"bad\name",
        "bad:name",
        "bad name",
        "bad%2Fname",
    ],
)
def test_store_rejects_path_control_resource_names(tmp_path, name):
    store = LocalStore(tmp_path / "store")

    with pytest.raises(InvalidPayloadError, match="Invalid channel"):
        store.create_channel({"name": name})

    store.create_channel({"name": "events"})

    with pytest.raises(InvalidPayloadError, match="Invalid event"):
        store.append_event({"id": name, "channel": "events", "payload": {}})

    with pytest.raises(InvalidPayloadError, match="Invalid subscription"):
        store.create_subscription("events", subscription_id=name)

    with pytest.raises(InvalidPayloadError, match="Invalid snapshot"):
        store.put_snapshot({"name": name, "channel": "events", "value": {}})


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        ".",
        "..",
        "bad/name",
        r"bad\name",
        "bad:name",
        "bad name",
        "bad%2Fname",
    ],
)
def test_store_rejects_path_control_lookup_names(tmp_path, name):
    store = LocalStore(tmp_path / "store")

    with pytest.raises(InvalidPayloadError, match="channel.name"):
        store.get_channel(name)
    with pytest.raises(InvalidPayloadError, match="channel.name"):
        store.query_events(name)
    with pytest.raises(InvalidPayloadError, match="event.id"):
        store.get_event(name)
    with pytest.raises(InvalidPayloadError, match="subscription.id"):
        store.get_subscription(name)
    with pytest.raises(InvalidPayloadError, match="subscription.id"):
        store.pull_subscription(name)
    with pytest.raises(InvalidPayloadError, match="artifact.id"):
        store.get_artifact(name)
    with pytest.raises(InvalidPayloadError, match="artifact.id"):
        store.read_artifact(name)
    with pytest.raises(InvalidPayloadError, match="snapshot.name"):
        store.get_snapshot(name)


def test_core_models_reject_non_string_resource_segments():
    required_values = (
        None,
        123,
        b"events",
        [],
        "   ",
        "bad name",
        "bad%2Fname",
    )
    optional_values = (123, b"events", [], "   ", "bad name", "bad%2Fname")

    for value in required_values:
        with pytest.raises(ValidationError):
            Channel(name=value)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Event(id=value, channel="events")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Event(id="event", channel=value)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Event(
                id="event",
                channel="events",
                parent_ids=(value,),  # type: ignore[list-item]
            )
        with pytest.raises(ValidationError):
            Artifact(id=value, path="artifacts/payload", size=1)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Artifact(
                id="artifact",
                path="artifacts/payload",
                size=1,
                event_ids=(value,),  # type: ignore[list-item]
            )

        with pytest.raises(ValidationError):
            Snapshot(name=value)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Subscription(id=value, channel="events")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Subscription(id="subscription", channel=value)  # type: ignore[arg-type]

    for value in optional_values:
        with pytest.raises(ValidationError):
            Artifact(
                id="artifact",
                channel=value,  # type: ignore[arg-type]
                path="artifacts/payload",
                size=1,
            )
        with pytest.raises(ValidationError):
            Snapshot(name="latest", channel=value)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Snapshot(name="latest", source_event_id=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kind",
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
def test_core_event_rejects_malformed_kind_tokens(kind):
    with pytest.raises(ValidationError):
        Event(channel="events", kind=kind)


@pytest.mark.parametrize(
    "value",
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
def test_core_models_reject_malformed_coordination_tokens(value):
    with pytest.raises(ValidationError):
        Event(channel="events", correlation_id=value)
    with pytest.raises(ValidationError):
        Subscription(channel="events", consumer=value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Channel(name="events", schema=b"demo.schemas:Event"),
        lambda: Channel(name="events", description=b"event stream"),
        lambda: Event(channel="events", source=b"sensor"),
        lambda: Event(channel="events", kind=b"raw"),
        lambda: Event(channel="events", schema=b"demo.schemas:Event"),
        lambda: Event(channel="events", correlation_id=b"corr"),
        lambda: Snapshot(name="latest", schema=b"demo.schemas:State"),
        lambda: Subscription(channel="events", consumer=b"worker"),
    ],
)
def test_core_models_reject_bytes_for_protocol_string_fields(factory):
    with pytest.raises(ValidationError):
        factory()


def test_event_append_and_query_with_metadata(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events", "schema": "demo.schemas:Event"})
    parent = store.append_event(
        Event(
            channel="events",
            source="test",
            kind="raw",
            payload={"value": 1},
            schema="demo.schemas:Event",
            metadata={"a": "b"},
            correlation_id="corr-1",
        )
    )
    child = store.append_event(
        {
            "channel": "events",
            "source": "test",
            "kind": "analysis",
            "payload": {"value": 2},
            "parent_ids": [parent.id],
        }
    )

    events = store.query_events("events")
    raw = store.query_events("events", kind="raw")
    after_parent = store.query_events("events", after_cursor=parent.cursor or 0)

    assert [event.id for event in events] == [parent.id, child.id]
    assert raw == (parent,)
    assert parent.cursor == 1
    assert child.cursor == 2
    assert store.get_event(parent.id) == parent
    assert [event.id for event in after_parent] == [child.id]
    assert child.parent_ids == (parent.id,)
    assert parent.metadata == {"a": "b"}
    with pytest.raises(ChannelNotFoundError):
        store.append_event({"channel": "missing", "payload": {}})
    with pytest.raises(EventNotFoundError):
        store.append_event(
            {"channel": "events", "payload": {}, "parent_ids": ["missing"]}
        )
    with pytest.raises(InvalidPayloadError, match="Invalid event"):
        store.append_event({"payload": {}})
    with pytest.raises(EventNotFoundError):
        store.get_event("missing")


@pytest.mark.parametrize("metadata", [[], [("owner", "tests")], "bad", 123])
def test_local_store_append_event_rejects_non_object_metadata(tmp_path, metadata):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})

    with pytest.raises(InvalidPayloadError, match="event.metadata"):
        store.append_event({"channel": "events", "payload": {}, "metadata": metadata})

    assert store.query_events("events") == ()


def test_query_events_validates_cursor_and_limit(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    store.append_event({"channel": "events", "payload": {"n": 1}})

    with pytest.raises(InvalidPayloadError, match="after_cursor"):
        store.query_events("events", after_cursor=-1)
    with pytest.raises(InvalidPayloadError, match="limit"):
        store.query_events("events", limit=0)


@pytest.mark.parametrize(
    "kind",
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
def test_store_rejects_malformed_event_kind_tokens(tmp_path, kind):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})

    with pytest.raises(InvalidPayloadError, match="event.kind"):
        store.append_event({"channel": "events", "kind": kind, "payload": {}})
    with pytest.raises(InvalidPayloadError, match="event.kind"):
        store.query_events("events", kind=kind)
    with pytest.raises(InvalidPayloadError, match="kind"):
        store.create_subscription("events", filters={"kind": kind})


@pytest.mark.parametrize(
    "value",
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
def test_store_rejects_malformed_coordination_tokens(tmp_path, value):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})

    with pytest.raises(InvalidPayloadError, match="event.correlation_id"):
        store.append_event(
            {"channel": "events", "correlation_id": value, "payload": {}}
        )
    with pytest.raises(InvalidPayloadError, match="subscription.consumer"):
        store.create_subscription("events", consumer=value)


def test_subscription_pull_advances_cursor(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    first = store.append_event({"channel": "events", "payload": {"n": 1}})
    second = store.append_event({"channel": "events", "payload": {"n": 2}})
    sub = store.create_subscription(
        "events",
        subscription_id="worker-events",
        consumer="worker",
        batch_size=1,
    )

    assert [event.id for event in store.pull_subscription(sub.id)] == [first.id]
    reused = store.create_subscription("events", subscription_id="worker-events")
    assert reused.id == sub.id
    assert reused.cursor == first.cursor
    store.create_channel({"name": "other-events"})
    with pytest.raises(SubscriptionExistsError):
        store.create_subscription("other-events", subscription_id="worker-events")
    assert [event.id for event in store.pull_subscription(sub.id)] == [second.id]
    assert store.get_subscription(sub.id).cursor == second.cursor
    assert store.pull_subscription(sub.id) == ()
    with pytest.raises(SubscriptionNotFoundError):
        store.pull_subscription("missing")


def test_subscription_pull_applies_kind_filter(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    store.append_event({"channel": "events", "kind": "raw", "payload": {"n": 1}})
    analysis = store.append_event(
        {"channel": "events", "kind": "analysis", "payload": {"n": 2}}
    )
    store.append_event({"channel": "events", "kind": "raw", "payload": {"n": 3}})
    sub = store.create_subscription("events", filters={"kind": "analysis"})

    assert [event.id for event in store.pull_subscription(sub.id)] == [analysis.id]
    assert store.pull_subscription(sub.id) == ()
    with pytest.raises(InvalidPayloadError, match="kind"):
        store.create_subscription("events", filters={"kind": 123})
    with pytest.raises(InvalidPayloadError, match="unsupported"):
        store.create_subscription("events", filters={"source": "sensor"})


def test_subscription_validates_batch_size_and_pull_limit(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "events"})
    sub = store.create_subscription("events")

    with pytest.raises(InvalidPayloadError, match="batch_size"):
        store.create_subscription("events", batch_size=0)
    for filters in ([], "kind", 123):
        with pytest.raises(InvalidPayloadError, match="subscription.filters"):
            store.create_subscription("events", filters=filters)  # type: ignore[arg-type]
    for metadata in ([], "bad", 123):
        with pytest.raises(InvalidPayloadError, match="subscription.metadata"):
            store.create_subscription("events", metadata=metadata)  # type: ignore[arg-type]
    with pytest.raises(InvalidPayloadError, match="limit"):
        store.pull_subscription(sub.id, limit=0)


def test_artifact_write_read_and_missing(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "artifacts", "form": "artifact-index"})
    event = store.append_event(
        {"channel": "artifacts", "payload": {"name": "greeting"}}
    )
    artifact = store.write_artifact(
        b"hello",
        channel="artifacts",
        media_type="text/plain",
        metadata={"name": "greeting"},
        event_ids=(event.id,),
    )

    assert artifact.size == 5
    assert artifact.sha256 is not None
    assert artifact.event_ids == (event.id,)
    assert store.get_artifact(artifact.id) == artifact
    assert store.read_artifact(artifact.id) == b"hello"
    with pytest.raises(ArtifactNotFoundError):
        store.read_artifact("missing")
    with pytest.raises(ArtifactNotFoundError):
        store.get_artifact("missing")
    with pytest.raises(EventNotFoundError):
        store.write_artifact(b"bad-link", event_ids=("missing",))


def test_artifact_write_rejects_non_byte_payloads(tmp_path):
    store = LocalStore(tmp_path / "store")

    for value in ("hello", 123, None):
        with pytest.raises(InvalidPayloadError, match="artifact data"):
            store.write_artifact(value)  # type: ignore[arg-type]

    payload = bytearray(b"hello")
    artifact = store.write_artifact(payload)  # type: ignore[arg-type]
    payload[:] = b"muted"

    assert artifact.size == 5
    assert store.read_artifact(artifact.id) == b"hello"
    memoryview_artifact = store.write_artifact(memoryview(b"world"))
    assert store.read_artifact(memoryview_artifact.id) == b"world"


def test_artifact_write_rejects_malformed_metadata_and_event_ids(tmp_path):
    store = LocalStore(tmp_path / "store")

    for metadata in ([], "bad", 123):
        with pytest.raises(InvalidPayloadError, match="artifact.metadata"):
            store.write_artifact(b"payload", metadata=metadata)  # type: ignore[arg-type]

    for event_ids in ("event-1", {"event-1": True}, 123, None):
        with pytest.raises(InvalidPayloadError, match="artifact.event_ids"):
            store.write_artifact(b"payload", event_ids=event_ids)  # type: ignore[arg-type]


def test_artifact_read_rejects_payload_paths_outside_store(tmp_path):
    store = LocalStore(tmp_path / "store")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    with sqlite3.connect(store.db_path) as db:
        db.execute(
            """
            insert into artifacts(id, channel, path, media_type, size, sha256, metadata, event_ids)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "escaped",
                None,
                "../outside.txt",
                "text/plain",
                outside.stat().st_size,
                None,
                "{}",
                "[]",
            ),
        )

    with pytest.raises(ArtifactNotFoundError, match="outside artifact storage"):
        store.read_artifact("escaped")


def test_snapshot_put_get_and_missing(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "state", "form": "latest-state"})
    event = store.append_event({"channel": "state", "payload": {"status": "ok"}})
    snapshot = store.put_snapshot(
        Snapshot(
            name="latest",
            channel="state",
            value={"status": "ok"},
            source_event_id=event.id,
        )
    )

    assert store.get_snapshot("latest") == snapshot
    updated = store.put_snapshot({"name": "latest", "value": {"status": "new"}})
    assert store.get_snapshot("latest").value == updated.value
    with pytest.raises(EventNotFoundError):
        store.put_snapshot(
            {"name": "dangling", "value": {"status": "bad"}, "source_event_id": "missing"}
        )
    with pytest.raises(SnapshotNotFoundError):
        store.get_snapshot("missing")


@pytest.mark.parametrize("metadata", [[], [("role", "latest")], "bad", 123])
def test_local_store_put_snapshot_rejects_non_object_metadata(tmp_path, metadata):
    store = LocalStore(tmp_path / "store")

    with pytest.raises(InvalidPayloadError, match="snapshot.metadata"):
        store.put_snapshot({"name": "latest", "value": {}, "metadata": metadata})

    with pytest.raises(SnapshotNotFoundError):
        store.get_snapshot("latest")
