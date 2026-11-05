"""Compose an LLLM network channel consumer with a local LLLM tactic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

try:
    from lllm import Tactic
except ImportError as exc:  # pragma: no cover - exercised in standalone installs
    raise RuntimeError("Install lllm to run this composition example.") from exc

from lllm.network import Event, LocalStore


class RawMessage(BaseModel):
    text: str


class AnalysisMessage(BaseModel):
    summary: str
    length: int


class AnalyzeMessage(Tactic[RawMessage, AnalysisMessage]):
    name = "analyze_message"
    input_type = RawMessage
    output_type = AnalysisMessage

    def _run(self, input_value, *, context=None):
        return AnalysisMessage(
            summary=input_value.text.upper(),
            length=len(input_value.text),
        )


def run_workflow(store: LocalStore):
    """Read one raw event through an LLLM tactic and write one analysis event."""

    store.create_channel(
        {"name": "raw", "schema": "demo.schemas:RawMessage", "form": "log"}
    )
    store.create_channel(
        {
            "name": "analysis",
            "schema": "demo.schemas:AnalysisMessage",
            "form": "log",
        }
    )
    subscription = store.create_subscription(
        "raw",
        consumer="lllm_tactic_processor",
        filters={"kind": "message"},
    )
    raw = store.append_event(
        Event(
            channel="raw",
            source="example",
            kind="message",
            payload={"text": "hello from the network"},
            correlation_id="episode-1",
        )
    )
    tactic = AnalyzeMessage()
    outputs = []
    for event in store.pull_subscription(subscription.id):
        result = tactic.run(event.payload)
        output = store.append_event(
            Event(
                channel="analysis",
                source="lllm_tactic_processor",
                kind="analysis",
                payload=result.model_dump(),
                correlation_id=event.correlation_id or event.id,
                parent_ids=(event.id,),
            )
        )
        outputs.append(output)

    return {
        "raw": raw,
        "analysis": tuple(outputs),
        "tactic_info": tactic.info(),
    }


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(prefix="lllm-network-tactic-"))
    result = run_workflow(LocalStore(root / ".lllm/network"))
    print(
        json.dumps(
            {
                "root": str(root),
                "raw_event_id": result["raw"].id,
                "analysis_event_ids": [event.id for event in result["analysis"]],
                "tactic": result["tactic_info"].name,
            },
            indent=2,
            sort_keys=True,
        )
    )
