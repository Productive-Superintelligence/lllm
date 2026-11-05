import asyncio

import httpx
import pytest
from pydantic import BaseModel

from lllm import Resolver, Tactic
from fastapi.testclient import TestClient

from lllm.network import AsyncRemoteStore, Event, LocalStore, RemoteStore, TacticProcessor
from lllm.network.server import create_app


SEEN_CONTEXTS = []


class Raw(BaseModel):
    text: str


class Summary(BaseModel):
    summary: str
    length: int


class Summarize(Tactic[Raw, Summary]):
    name = "summarize"
    input_type = Raw
    output_type = Summary

    def _run(self, input_value, *, context=None):
        if input_value.text == "boom":
            raise RuntimeError("cannot summarize boom")
        if context is not None and "event_id" in context.metadata:
            SEEN_CONTEXTS.append(context)
        return Summary(summary=input_value.text.upper(), length=len(input_value.text))


def make_store(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "raw", "form": "log"})
    store.create_channel({"name": "summaries", "form": "log"})
    return store


def append(store, text, **fields):
    return store.append_event(
        Event(channel="raw", source="test", kind="message", payload={"text": text}, **fields)
    )


def test_processor_writes_traceable_outputs_and_resumes(tmp_path):
    store = make_store(tmp_path)
    first = append(store, "hello", correlation_id="episode-1")
    processor = TacticProcessor(Summarize(), store, source="raw", sink="summaries")

    [output] = processor.process()

    assert output.payload == {"summary": "HELLO", "length": 5}
    assert output.parent_ids == (first.id,)
    assert output.correlation_id == "episode-1"
    assert output.source == "summarize-processor"
    context = SEEN_CONTEXTS[-1]
    assert context.metadata["event_id"] == first.id
    assert context.metadata["correlation_id"] == "episode-1"
    assert context.tags == {"channel": "raw"}
    assert processor.process() == []

    append(store, "again")
    resumed = TacticProcessor(Summarize(), store, source="raw", sink="summaries")
    assert [event.payload["summary"] for event in resumed.process()] == ["AGAIN"]


def test_processor_without_sink_returns_tactic_outputs(tmp_path):
    store = make_store(tmp_path)
    append(store, "one")
    append(store, "two")

    outputs = TacticProcessor(Summarize(), store, source="raw").process(limit=1)

    assert outputs == [Summary(summary="ONE", length=3)]


def test_processor_kind_filter_skips_other_events(tmp_path):
    store = make_store(tmp_path)
    store.append_event(Event(channel="raw", source="t", kind="noise", payload={"text": "x"}))
    append(store, "kept")

    outputs = TacticProcessor(Summarize(), store, source="raw", kind="message").process()

    assert outputs == [Summary(summary="KEPT", length=4)]


def test_processor_errors_raise_without_skipping_later_events(tmp_path):
    store = make_store(tmp_path)
    append(store, "boom")
    append(store, "after")
    processor = TacticProcessor(Summarize(), store, source="raw", sink="summaries")

    with pytest.raises(RuntimeError, match="boom"):
        processor.process()

    assert [event.payload["summary"] for event in processor.process()] == ["AFTER"]


def test_processor_can_record_errors_as_events(tmp_path):
    store = make_store(tmp_path)
    append(store, "boom")
    append(store, "fine")

    events = TacticProcessor(
        Summarize(), store, source="raw", sink="summaries", errors="record"
    ).process()

    assert [event.kind for event in events] == ["error", "result"]
    assert events[0].payload == {"error": "cannot summarize boom", "type": "RuntimeError"}


def test_processor_rejects_unknown_error_mode(tmp_path):
    with pytest.raises(ValueError, match="errors"):
        TacticProcessor(Summarize(), make_store(tmp_path), source="raw", errors="ignore")


def test_processor_runs_against_a_remote_store(tmp_path):
    local = make_store(tmp_path)
    append(local, "remote")
    app_client = TestClient(create_app(local))

    def forward(request):
        response = app_client.request(
            request.method,
            request.url.path,
            params=request.url.params,
            content=request.content,
            headers={"content-type": request.headers.get("content-type", "")},
        )
        return httpx.Response(response.status_code, content=response.content)

    remote = RemoteStore("http://testserver", transport=httpx.MockTransport(forward))

    outputs = TacticProcessor(Summarize(), remote, source="raw", sink="summaries").process()

    assert [event.payload["summary"] for event in outputs] == ["REMOTE"]
    assert [event.payload["summary"] for event in local.query_events("summaries")] == [
        "REMOTE"
    ]


def test_processor_runs_async_against_an_async_remote_store(tmp_path):
    local = make_store(tmp_path)
    append(local, "async")

    async def run():
        remote = AsyncRemoteStore(
            "http://testserver", transport=httpx.ASGITransport(app=create_app(local))
        )
        processor = TacticProcessor(Summarize(), remote, source="raw", sink="summaries")
        return await processor.aprocess()

    outputs = asyncio.run(run())

    assert [event.payload["summary"] for event in outputs] == ["ASYNC"]


def test_resolver_hands_back_tactics_and_stores_from_one_config(tmp_path):
    (tmp_path / ".lllm").mkdir()
    (tmp_path / ".lllm" / "config.toml").write_text(
        """
[refs."lllm://demo/briefs/tactics/summarize"]
url = "http://127.0.0.1:8123"

[refs."lllm://demo/briefs/channels/raw"]
store = "data/store"

[settings]
region = "us"
""",
        encoding="utf-8",
    )
    resolver = Resolver.from_config(tmp_path)

    tactic = resolver.resolve("lllm://demo/briefs/tactics/summarize")
    store = resolver.resolve("lllm://demo/briefs/channels/raw")

    assert tactic.__class__.__name__ == "RemoteTactic"
    assert isinstance(store, LocalStore)
    assert resolver.setting("region") == "us"
    assert resolver.refs() == (
        "lllm://demo/briefs/channels/raw",
        "lllm://demo/briefs/tactics/summarize",
    )
    with pytest.raises(ValueError, match="not /schemas/"):
        resolver.resolve("lllm://demo/briefs/schemas/raw")


def test_resolver_registers_local_tactics():
    resolver = Resolver()
    resolver.register("lllm://demo/briefs/tactics/summarize", Summarize())

    result = resolver.run("lllm://demo/briefs/tactics/summarize", {"text": "hi"})

    assert result == Summary(summary="HI", length=2)
