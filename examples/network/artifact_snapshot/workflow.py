from lllm.network import LocalStore, Snapshot


def run_workflow(store: LocalStore):
    """Write an event, attach an artifact, and update a latest-state snapshot."""

    store.create_channel(
        {
            "name": "events",
            "schema": "demo.schemas:Event",
            "form": "log",
            "description": "Semantic event log.",
        }
    )
    store.create_channel(
        {
            "name": "frames",
            "form": "artifact-index",
            "description": "Artifacts linked to events.",
        }
    )
    event = store.append_event(
        {
            "channel": "events",
            "source": "artifact_snapshot_example",
            "kind": "frame",
            "payload": {"frame": 1},
            "correlation_id": "episode-1",
        }
    )
    artifact = store.write_artifact(
        b"frame-one",
        channel="frames",
        media_type="text/plain",
        metadata={"role": "observation"},
        event_ids=(event.id,),
    )
    snapshot = store.put_snapshot(
        Snapshot(
            name="latest",
            channel="events",
            value={
                "last_event_id": event.id,
                "artifact_id": artifact.id,
                "frame": 1,
            },
            schema="demo.schemas:LatestState",
            source_event_id=event.id,
            metadata={"strategy": "latest-state"},
        )
    )
    return {
        "event": event,
        "artifact": artifact,
        "artifact_data": store.read_artifact(artifact.id),
        "snapshot": snapshot,
    }
