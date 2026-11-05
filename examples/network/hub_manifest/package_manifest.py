from typing import Any

from lllm.network import Channel, Snapshot, channel_resource, endpoint, snapshot_resource


@endpoint.get(
    "/channels/{name}/tail",
    scope="channel",
    description="Return the most recent events for a channel.",
    tags=("channels",),
)
def channel_tail(store, name: str, limit: int = 20):
    return store.query_events(name, limit=limit)


@endpoint.get(
    "/snapshots/latest-analysis",
    scope="snapshot",
    description="Return the latest derived analysis state.",
    tags=("snapshots",),
)
def latest_analysis(store):
    return store.get_snapshot("latest_analysis")


RAW_CHANNEL = Channel(
    name="raw",
    schema="raw_event",
    form="log",
    description="Incoming events from producers.",
    metadata={"retention": "local"},
)
ANALYSIS_CHANNEL = Channel(
    name="analysis",
    schema="analysis_event",
    form="log",
    description="Derived analysis events.",
    metadata={"derived_from": "raw"},
)
LATEST_ANALYSIS = Snapshot(
    name="latest_analysis",
    channel="analysis",
    schema="analysis_event",
    metadata={"derived_from": "analysis"},
)


def build_channel_resources() -> dict[str, dict[str, Any]]:
    resources = [
        channel_resource(RAW_CHANNEL, custom_endpoints=[channel_tail]),
        channel_resource(ANALYSIS_CHANNEL),
    ]
    return {
        resource["name"]: {
            key: value
            for key, value in resource.items()
            if key != "name" and value not in (None, "", {}, [])
        }
        for resource in resources
    }


def build_snapshot_resources() -> dict[str, dict[str, Any]]:
    resource = snapshot_resource(
        LATEST_ANALYSIS,
        description="Latest derived analysis event.",
        custom_endpoints=[latest_analysis],
    )
    return {
        resource["name"]: {
            key: value
            for key, value in resource.items()
            if key != "name" and value not in (None, "", {}, [])
        }
    }


def build_manifest() -> dict[str, Any]:
    return {
        "package": {
            "manifest_version": "0.1",
            "org": "demo",
            "name": "semantic-events",
            "version": "0.1.0",
            "kind": "channel",
            "primary": "channels.raw",
            "description": "LLLM network channel package for raw and analysis events.",
        },
        "schemas": {
            "raw_event": {"entry": "demo.schemas:RawEvent"},
            "analysis_event": {"entry": "demo.schemas:AnalysisEvent"},
        },
        "channels": build_channel_resources(),
        "snapshots": build_snapshot_resources(),
        "runs": {
            "local": {
                "channels": ["raw", "analysis"],
                "snapshots": ["latest_analysis"],
                "description": "Create both channels in a local store.",
            }
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_manifest(), indent=2))
