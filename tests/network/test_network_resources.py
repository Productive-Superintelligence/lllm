import pytest

from lllm.network import Channel, Snapshot
from lllm.network.resources import channel_resource, snapshot_resource
from lllm.network.server import endpoint


def test_channel_resource_exports_metadata_and_custom_endpoints():
    @endpoint.get("/channels/events/range", scope="channel", tags=("events",))
    def event_range(store, body=None):
        return []

    resource = channel_resource(
        Channel(
            name="events",
            schema="demo.schemas:Event",
            form="log",
            description="Event log.",
            metadata={"retention": "local"},
        ),
        custom_endpoints=[event_range],
    )

    assert resource == {
        "name": "events",
        "schema": "demo.schemas:Event",
        "form": "log",
        "description": "Event log.",
        "metadata": {"retention": "local"},
        "endpoints": [
            {
                "name": "event_range",
                "method": "GET",
                "path": "/channels/events/range",
                "scope": "channel",
                "description": "",
                "tags": ["events"],
            }
        ],
    }


def test_channel_resource_exports_full_custom_endpoint_methods():
    @endpoint.put("/channels/events/state", scope="channel", tags=("events",))
    def put_state(store, body=None):
        return body

    @endpoint.patch(
        "/channels/events/state",
        scope="channel",
        description="Patch channel state.",
        tags=("events",),
    )
    def patch_state(store, body=None):
        return body

    @endpoint.delete("/channels/events/state", scope="channel", tags=("events",))
    def delete_state(store):
        return {"deleted": True}

    resource = channel_resource(
        Channel(name="events"),
        custom_endpoints=[put_state, patch_state, delete_state],
    )

    assert resource["endpoints"] == [
        {
            "name": "put_state",
            "method": "PUT",
            "path": "/channels/events/state",
            "scope": "channel",
            "description": "",
            "tags": ["events"],
        },
        {
            "name": "patch_state",
            "method": "PATCH",
            "path": "/channels/events/state",
            "scope": "channel",
            "description": "Patch channel state.",
            "tags": ["events"],
        },
        {
            "name": "delete_state",
            "method": "DELETE",
            "path": "/channels/events/state",
            "scope": "channel",
            "description": "",
            "tags": ["events"],
        },
    ]


def test_channel_resource_rejects_ambiguous_custom_endpoint_metadata():
    @endpoint.get("/channels/events/range", name="shared")
    def first_range(store):
        return []

    @endpoint.post("/channels/events/publish", name="shared")
    def second_range(store):
        return []

    with pytest.raises(ValueError, match="duplicate custom endpoint name"):
        channel_resource(
            Channel(name="events"),
            custom_endpoints=[first_range, second_range],
        )

    @endpoint.get("/channels/{name}/tail", name="tail_by_name")
    def tail_by_name(store, name: str):
        return []

    @endpoint.get("/channels/{channel}/tail", name="tail_by_channel")
    def tail_by_channel(store, channel: str):
        return []

    with pytest.raises(ValueError, match="duplicate custom endpoint route"):
        channel_resource(
            Channel(name="events"),
            custom_endpoints=[tail_by_name, tail_by_channel],
        )


def test_channel_resource_isolates_nested_metadata():
    channel = Channel(name="events", metadata={"labels": ["raw"]})
    resource = channel_resource(channel)

    channel.metadata["labels"].append("source-mutated")
    resource["metadata"]["labels"].append("resource-mutated")

    assert channel.metadata == {"labels": ["raw", "source-mutated"]}
    assert channel_resource(channel)["metadata"] == {"labels": ["raw", "source-mutated"]}
    assert resource["metadata"] == {"labels": ["raw", "resource-mutated"]}


def test_snapshot_resource_exports_metadata_and_custom_endpoints():
    @endpoint.get(
        "/snapshots/latest",
        scope="snapshot",
        description="Read the latest state.",
        tags=("snapshots",),
    )
    def latest_state(store):
        return store.get_snapshot("latest")

    resource = snapshot_resource(
        Snapshot(
            name="latest",
            channel="analysis",
            schema="demo.schemas:Analysis",
            value={"summary": "ok"},
            source_event_id="event-1",
            metadata={"retention": "latest"},
        ),
        description="Latest analysis state.",
        custom_endpoints=[latest_state],
    )

    assert resource == {
        "name": "latest",
        "schema": "demo.schemas:Analysis",
        "channel": "analysis",
        "description": "Latest analysis state.",
        "metadata": {"retention": "latest"},
        "endpoints": [
            {
                "name": "latest_state",
                "method": "GET",
                "path": "/snapshots/latest",
                "scope": "snapshot",
                "description": "Read the latest state.",
                "tags": ["snapshots"],
            }
        ],
    }


def test_snapshot_resource_isolates_nested_metadata():
    snapshot = Snapshot(
        name="latest",
        value={"summary": "ok"},
        metadata={"labels": ["latest"]},
    )
    resource = snapshot_resource(snapshot)

    snapshot.metadata["labels"].append("source-mutated")
    resource["metadata"]["labels"].append("resource-mutated")

    assert snapshot.metadata == {"labels": ["latest", "source-mutated"]}
    assert snapshot_resource(snapshot)["metadata"] == {
        "labels": ["latest", "source-mutated"]
    }
    assert resource["metadata"] == {"labels": ["latest", "resource-mutated"]}
