"""Run a tactic over channel events with TacticProcessor."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from lllm import Tactic
from lllm.network import Event, LocalStore, TacticProcessor


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
    """Feed raw events through a tactic and write analysis events."""

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
    processor = TacticProcessor(
        AnalyzeMessage(),
        store,
        source="raw",
        sink="analysis",
        consumer="lllm_tactic_processor",
        kind="message",
        output_kind="analysis",
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
    outputs = processor.process()

    return {
        "raw": raw,
        "analysis": tuple(outputs),
        "tactic_info": processor.tactic.info(),
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
