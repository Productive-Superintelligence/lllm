from lllm.network import Event, LocalStore


def process_once(
    store: LocalStore,
    *,
    subscription_id: str,
    output_channel: str = "analysis",
) -> tuple[Event, ...]:
    """Read pending raw events and write one analysis event per input event."""

    outputs: list[Event] = []
    for event in store.pull_subscription(subscription_id):
        output = store.append_event(
            Event(
                channel=output_channel,
                source="channel_processor",
                kind="analysis",
                payload={
                    "strategy": "baseline",
                    "summary": str(event.payload),
                    "source_event_id": event.id,
                },
                correlation_id=event.correlation_id or event.id,
                parent_ids=(event.id,),
            )
        )
        outputs.append(output)
    return tuple(outputs)
