from lllm.network import Channel, LocalStore


def run_workflow(store: LocalStore):
    """Create a local channel, append one event, and read it back."""

    channel = store.create_channel(
        Channel(
            name="events",
            schema="demo.schemas:Event",
            form="log",
            description="Local event stream.",
        )
    )
    event = store.append_event(
        {
            "channel": channel.name,
            "kind": "message",
            "payload": {"text": "hello"},
            "schema": channel.schema,
            "source": "first_channel_example",
        }
    )
    events = store.query_events(channel.name, after_cursor=0)
    return {
        "channel": channel,
        "event": event,
        "events": events,
    }
