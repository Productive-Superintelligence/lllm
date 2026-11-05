import asyncio

import httpx
import pytest
from pydantic import BaseModel, field_serializer
from pydantic_core import PydanticSerializationError

from lllm import Resolver, Tactic, as_tactic
from fastapi.testclient import TestClient

from lllm.network import (
    AsyncRemoteStore,
    ChannelNotFoundError,
    Event,
    InvalidPayloadError,
    LocalStore,
    RemoteStore,
    TacticProcessor,
)
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
    assert output.source == "summarize-raw-processor"
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

    # The failing event stays unacknowledged, so the next call retries it
    # rather than skipping ahead to "after".
    with pytest.raises(RuntimeError, match="boom"):
        processor.process()
    assert store.query_events("summaries") == ()
    assert store.get_subscription("summarize-raw-processor").cursor == 0


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


class Flaky(Tactic[Raw, Summary]):
    """Fail the first ``failures`` runs, then summarize like ``Summarize``."""

    name = "flaky"
    input_type = Raw
    output_type = Summary

    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def _run(self, input_value, *, context=None):
        if self.failures:
            self.failures -= 1
            raise RuntimeError(f"transient failure on {input_value.text}")
        return Summary(summary=input_value.text.upper(), length=len(input_value.text))


def make_remote(local):
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

    return RemoteStore("http://testserver", transport=httpx.MockTransport(forward))


def summaries(events):
    return [event.payload["summary"] for event in events]


def test_processor_default_consumer_uses_the_tactic_instance_name(tmp_path):
    store = make_store(tmp_path)
    append(store, "Hi")

    def shout(payload: dict) -> dict:
        return {"text": payload["text"].upper()}

    def whisper(payload: dict) -> dict:
        return {"text": payload["text"].lower()}

    loud = TacticProcessor(as_tactic(shout), store, source="raw")
    quiet = TacticProcessor(as_tactic(whisper), store, source="raw")
    renamed = TacticProcessor(Summarize(name="digest"), store, source="raw")

    assert loud.consumer == "shout-raw-processor"
    assert quiet.consumer == "whisper-raw-processor"
    assert renamed.consumer == "digest-raw-processor"
    # Distinct subscriptions, so every processor sees the event.
    assert loud.process() == [{"text": "HI"}]
    assert quiet.process() == [{"text": "hi"}]
    assert renamed.process() == [Summary(summary="HI", length=2)]


def test_processor_rejects_a_consumer_subscribed_to_another_channel(tmp_path):
    store = make_store(tmp_path)
    store.create_channel({"name": "other", "form": "log"})
    TacticProcessor(Summarize(), store, source="raw", consumer="shared").process()
    other = TacticProcessor(Summarize(), store, source="other", consumer="shared")

    with pytest.raises(ValueError, match="'raw'.*'other'.*consumer="):
        other.process()
    with pytest.raises(ValueError, match="consumer="):
        asyncio.run(other.aprocess())


def test_processor_rejects_a_consumer_with_another_kind_filter(tmp_path):
    store = make_store(tmp_path)

    def processor(kind):
        return TacticProcessor(
            Summarize(), store, source="raw", consumer="shared", kind=kind
        )

    processor("message").process()

    for kind in ("note", None):
        with pytest.raises(ValueError, match="kind filter 'message'.*consumer="):
            processor(kind).process()

    append(store, "again")
    assert processor("message").process() == [Summary(summary="AGAIN", length=5)]


def test_processor_retries_a_failed_event_on_the_next_call(tmp_path):
    store = make_store(tmp_path)
    store.append_event(Event(channel="raw", source="t", kind="noise", payload={"text": "x"}))
    failing = append(store, "boom")
    append(store, "after")
    processor = TacticProcessor(
        Flaky(failures=2), store, source="raw", sink="summaries", kind="message"
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="transient failure on boom"):
            processor.process()
    assert store.get_subscription("flaky-raw-message-processor").cursor == 0
    assert store.query_events("summaries") == ()

    outputs = processor.process()

    assert summaries(outputs) == ["BOOM", "AFTER"]
    assert outputs[0].parent_ids == (failing.id,)
    assert summaries(store.query_events("summaries")) == ["BOOM", "AFTER"]
    assert processor.process() == []


def test_processor_retries_failed_events_through_remote_stores(tmp_path):
    local = make_store(tmp_path / "sync")
    append(local, "boom")
    append(local, "after")
    processor = TacticProcessor(
        Flaky(failures=1), make_remote(local), source="raw", sink="summaries"
    )

    with pytest.raises(RuntimeError, match="boom"):
        processor.process()
    assert summaries(processor.process()) == ["BOOM", "AFTER"]

    async_local = make_store(tmp_path / "async")
    append(async_local, "boom")
    append(async_local, "after")

    async def run():
        transport = httpx.ASGITransport(app=create_app(async_local))
        remote = AsyncRemoteStore("http://testserver", transport=transport)
        processor = TacticProcessor(
            Flaky(failures=1), remote, source="raw", sink="summaries"
        )
        with pytest.raises(RuntimeError, match="boom"):
            await processor.aprocess()
        return await processor.aprocess()

    assert summaries(asyncio.run(run())) == ["BOOM", "AFTER"]
    assert summaries(async_local.query_events("summaries")) == ["BOOM", "AFTER"]


def test_processor_record_mode_acknowledges_the_failed_event(tmp_path):
    store = make_store(tmp_path)
    failing = append(store, "boom")
    append(store, "fine")
    processor = TacticProcessor(
        Flaky(failures=1), store, source="raw", sink="summaries", errors="record"
    )

    events = processor.process()

    assert [event.kind for event in events] == ["error", "result"]
    assert events[0].parent_ids == (failing.id,)
    # Recorded, so acknowledged: the event is not handed to the tactic again.
    assert processor.process() == []


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_runs_the_event_its_acknowledgment_took_from_another_worker(
    tmp_path, mode
):
    store = make_store(tmp_path)
    append(store, "first")
    append(store, "second")

    class Raced(Summarize):
        def _run(self, input_value, *, context=None):
            if input_value.text == "first":
                # Another worker on this subscription acknowledges "first" meanwhile.
                store.pull_subscription("summarize-raw-processor", limit=1)
            return super()._run(input_value, context=context)

    processor = TacticProcessor(Raced(), store, source="raw", sink="summaries")
    outputs = processor.process() if mode == "sync" else asyncio.run(processor.aprocess())

    # The output for "first" is written before acknowledging it shows the other
    # worker had it. That acknowledgment took "second", which is still
    # processed, exactly once.
    assert summaries(outputs) == ["FIRST", "SECOND"]
    assert summaries(store.query_events("summaries")) == ["FIRST", "SECOND"]
    assert processor.process() == []


def test_processor_process_rejects_async_stores_with_a_clear_error(tmp_path):
    local = make_store(tmp_path)
    append(local, "async")
    remote = AsyncRemoteStore(
        "http://testserver", transport=httpx.ASGITransport(app=create_app(local))
    )
    processor = TacticProcessor(Summarize(), remote, source="raw", sink="summaries")

    with pytest.raises(TypeError, match=r"AsyncRemoteStore.*aprocess\(\)"):
        processor.process()

    # The rejected call left no half-made state behind.
    assert summaries(asyncio.run(processor.aprocess())) == ["ASYNC"]


class Fragile(BaseModel):
    """A valid output whose JSON dump fails for the text "binary"."""

    text: str

    @field_serializer("text")
    def _refuse_binary(self, value):
        if value == "binary":
            raise ValueError("cannot serialize binary")
        return value


class Echo(Tactic[Raw, Fragile]):
    name = "echo"
    input_type = Raw
    output_type = Fragile

    def _run(self, input_value, *, context=None):
        return Fragile(text=input_value.text)


def runner(processor, mode):
    if mode == "sync":
        return processor.process
    return lambda: asyncio.run(processor.aprocess())


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_records_outputs_that_cannot_become_sink_events(tmp_path, mode):
    store = make_store(tmp_path)
    failing = append(store, "binary")
    append(store, "hello")
    append(store, "world")
    processor = TacticProcessor(
        Echo(), store, source="raw", sink="summaries", errors="record"
    )
    run = runner(processor, mode)

    events = run()

    # The failed output is recorded and acknowledged, so it does not block the
    # events behind it.
    assert [event.kind for event in events] == ["error", "result", "result"]
    assert events[0].parent_ids == (failing.id,)
    assert "cannot serialize binary" in events[0].payload["error"]
    assert [event.payload for event in events[1:]] == [{"text": "hello"}, {"text": "world"}]
    assert store.get_subscription("echo-raw-processor").cursor == 3
    assert run() == []


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_retries_outputs_that_cannot_become_sink_events(tmp_path, mode):
    store = make_store(tmp_path)
    append(store, "binary")
    append(store, "hello")
    run = runner(TacticProcessor(Echo(), store, source="raw", sink="summaries"), mode)

    for _ in range(2):
        with pytest.raises(PydanticSerializationError, match="cannot serialize binary"):
            run()
    assert store.get_subscription("echo-raw-processor").cursor == 0
    assert store.query_events("summaries") == ()

    # The same subscription picks up at the failed event, not after it.
    recording = TacticProcessor(
        Echo(), store, source="raw", sink="summaries", errors="record"
    )
    assert [event.kind for event in recording.process()] == ["error", "result"]


@pytest.mark.parametrize("name", ["shout loud", "tab\tname", "ask?", "a#b"])
def test_processor_needs_a_consumer_for_tactic_names_that_cannot_name_one(
    tmp_path, name
):
    store = make_store(tmp_path)
    append(store, "Hi")

    def shout(payload: dict) -> dict:
        return {"text": payload["text"].upper()}

    # Valid tactic names, but whitespace is not valid in a subscription id and
    # "?" or "#" would cut a RemoteStore URL short.
    for tactic in (as_tactic(shout, name=name), Summarize(name=name)):
        with pytest.raises(ValueError, match=r"default consumer .*Pass consumer="):
            TacticProcessor(tactic, store, source="raw")

    named = TacticProcessor(
        as_tactic(shout, name=name), store, source="raw", consumer="shout"
    )
    assert named.process() == [{"text": "HI"}]


class Opaque:
    """An output no store can serialize."""


class Unstorable(Tactic):
    name = "unstorable"

    def _run(self, input_value, *, context=None):
        if input_value["text"] == "bad":
            return Opaque()
        return {"text": input_value["text"]}


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_records_outputs_the_store_rejects(tmp_path, mode):
    store = make_store(tmp_path)
    failing = append(store, "bad")
    append(store, "good")
    processor = TacticProcessor(
        Unstorable(), store, source="raw", sink="summaries", errors="record"
    )
    run = runner(processor, mode)

    events = run()

    # The rejected append is recorded like a tactic failure, before the event
    # is acknowledged, so it neither raises nor blocks the event behind it.
    assert [event.kind for event in events] == ["error", "result"]
    assert events[0].parent_ids == (failing.id,)
    assert events[0].payload["type"] == "InvalidPayloadError"
    assert events[1].payload == {"text": "good"}
    assert [event.id for event in store.query_events("summaries")] == [
        event.id for event in events
    ]
    assert run() == []


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_retries_outputs_the_store_rejects(tmp_path, mode):
    store = make_store(tmp_path)
    append(store, "bad")
    append(store, "good")
    processor = TacticProcessor(Unstorable(), store, source="raw", sink="summaries")
    run = runner(processor, mode)

    # Not acknowledged, so the next call runs the same event again rather than
    # skipping ahead to "good".
    for _ in range(2):
        with pytest.raises(InvalidPayloadError, match="not JSON serializable"):
            run()
    assert store.get_subscription("unstorable-raw-processor").cursor == 0
    assert store.query_events("summaries") == ()


@pytest.mark.parametrize("errors", ["raise", "record"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_processor_keeps_events_whose_sink_write_failed(tmp_path, errors, mode):
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "raw", "form": "log"})
    append(store, "hello")
    processor = TacticProcessor(
        Summarize(), store, source="raw", sink="summaries", errors=errors
    )
    run = runner(processor, mode)

    # Without the sink channel neither the output nor an error event can be
    # written, so the event stays unacknowledged instead of being lost.
    with pytest.raises(ChannelNotFoundError):
        run()
    assert store.get_subscription("summarize-raw-processor").cursor == 0

    store.create_channel({"name": "summaries", "form": "log"})
    assert summaries(run()) == ["HELLO"]


def test_processor_default_consumer_is_per_source_and_kind(tmp_path):
    store = make_store(tmp_path)
    store.create_channel({"name": "other", "form": "log"})
    append(store, "raw")
    store.append_event(
        Event(channel="other", source="test", kind="message", payload={"text": "other"})
    )

    on_raw = TacticProcessor(Summarize(), store, source="raw")
    on_other = TacticProcessor(Summarize(), store, source="other")
    messages = TacticProcessor(Summarize(), store, source="raw", kind="message")

    assert on_raw.consumer == "summarize-raw-processor"
    assert on_other.consumer == "summarize-other-processor"
    assert messages.consumer == "summarize-raw-message-processor"
    # One subscription each, so every processor reads its own events.
    assert on_raw.process() == [Summary(summary="RAW", length=3)]
    assert on_other.process() == [Summary(summary="OTHER", length=5)]
    assert messages.process() == [Summary(summary="RAW", length=3)]


@pytest.mark.parametrize("fields", [{"source": "in?box"}, {"kind": "a#b"}])
def test_processor_needs_a_consumer_for_sources_and_kinds_that_cannot_name_one(
    tmp_path, fields
):
    store = make_store(tmp_path)
    fields = {"source": "raw", **fields}

    with pytest.raises(ValueError, match=r"default consumer .*Pass consumer="):
        TacticProcessor(Summarize(), store, **fields)
    named = TacticProcessor(Summarize(), store, consumer="digest", **fields)
    assert named.subscription_id == "digest"
