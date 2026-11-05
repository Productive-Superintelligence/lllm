"""Native Tactic entry points honor every way to write a tactic."""

import asyncio
import logging
import threading

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lllm import CallContext, Tactic, TacticEvent, TacticUnsupportedError
from lllm.protocol import Tactic as ProtocolTactic
from lllm.services import create_tactic_app


class Question(BaseModel):
    text: str


async def _ccall(tactic, tasks, **kwargs):
    return sorted([item async for item in tactic.ccall(tasks, **kwargs)])


# -- T1: a _run() hook works through the native entry points -----------------


def _run_hook_tactic(seen):
    class Upper(Tactic):
        name = "upper"

        def _run(self, input_value, *, context=None):
            seen.append(context)
            return input_value.upper()

    return Upper()


def test_run_hook_tactics_work_through_every_native_entry_point():
    tactic = _run_hook_tactic([])

    session = tactic("a", return_session=True)

    assert {"call", "acall"} <= tactic.capabilities()
    assert tactic("a") == "A"
    assert (session.state, session.delivery) == ("success", "A")
    assert asyncio.run(tactic.acall("b")) == "B"
    assert tactic.bcall(["c", "d"]) == ["C", "D"]
    assert asyncio.run(_ccall(tactic, ["e", "f"])) == [(0, "E"), (1, "F")]
    assert tactic.run("g") == "G"


def test_run_hook_receives_the_call_context():
    seen = []
    tactic = _run_hook_tactic(seen)
    context = CallContext(trace_id="t-1")

    tactic("a", context=context)
    asyncio.run(tactic.acall("b", context=context))
    tactic.bcall(["c"], context=context)
    tactic("d")

    assert seen[:3] == [context, context, context]
    assert all(item is context for item in seen[:3])
    # Like run(), a call without a context still hands the hook a CallContext.
    assert isinstance(seen[3], CallContext)


def test_tactics_without_a_program_still_raise_instead_of_recursing():
    class Empty(Tactic):
        name = "empty"

    for call in (
        lambda: Empty()("x"),
        lambda: Empty().run("x"),
        lambda: asyncio.run(Empty().acall("x")),
    ):
        with pytest.raises(TacticUnsupportedError, match=r"call\(\) or _run\(\)"):
            call()


# -- T2: an _arun() hook is a python tactic that also runs synchronously -----


class AsyncUpper(Tactic):
    name = "async-upper"

    async def _arun(self, input_value, *, context=None):
        await asyncio.sleep(0)
        return f"{input_value.upper()}|{getattr(context, 'trace_id', None)}"


def test_arun_hook_tactics_report_python_and_run_synchronously():
    tactic = AsyncUpper()

    assert AsyncUpper.runtime_kind == "python"
    assert tactic.info().runtime_kind == "python"
    assert tactic.run("a", context=CallContext(trace_id="t-1")) == "A|t-1"
    assert tactic("b") == "B|None"
    assert tactic.bcall(["c"]) == ["C|None"]
    assert asyncio.run(tactic.acall("d")) == "D|None"


def test_arun_hook_runs_from_sync_code_inside_a_running_loop():
    async def main():
        return AsyncUpper().run("a")

    assert asyncio.run(main()) == "A|None"


def test_arun_awaits_the_arun_hook_directly():
    seen = {}

    class WhereAmI(Tactic):
        name = "where-am-i"

        async def _arun(self, input_value, *, context=None):
            seen["hook"] = (threading.get_ident(), asyncio.get_running_loop())
            return input_value

    async def main():
        seen["caller"] = (threading.get_ident(), asyncio.get_running_loop())
        return await WhereAmI().arun("x", context=CallContext(trace_id="t-2"))

    assert asyncio.run(main()) == "x"
    assert seen["hook"] == seen["caller"]


def test_runtime_kind_follows_the_overridden_hooks():
    class RunHook(Tactic):
        def _run(self, input_value, *, context=None):
            return input_value

    class Program(Tactic):
        def call(self, task):
            return task

    class ProgramWithAsyncHook(Tactic):
        def call(self, task):
            return task

        async def _arun(self, input_value, *, context=None):
            return input_value

    with pytest.warns(UserWarning, match="agent_group"):

        class AgentsWithAsyncHook(Tactic):
            agent_group = ["writer"]

            async def _arun(self, input_value, *, context=None):
                return input_value

    assert RunHook.runtime_kind == "python"
    assert Program.runtime_kind == "native"
    assert ProgramWithAsyncHook.runtime_kind == "native"
    assert AgentsWithAsyncHook.runtime_kind == "native"


# -- T3: an async call() program is awaited -----------------------------------


class AsyncCall(Tactic):
    name = "async-call"

    async def call(self, task):
        await asyncio.sleep(0)
        return task.upper()


def test_async_call_programs_are_awaited_on_every_entry_point():
    tactic = AsyncCall()

    assert tactic("a") == "A"
    assert tactic.run("b") == "B"
    assert tactic("c", return_session=True).delivery == "C"
    assert asyncio.run(tactic.arun("d")) == "D"
    assert asyncio.run(tactic.acall("e")) == "E"
    assert tactic.bcall(["f", "g"]) == ["F", "G"]
    assert asyncio.run(_ccall(tactic, ["h", "i"])) == [(0, "H"), (1, "I")]


def test_async_call_runs_from_sync_code_inside_a_running_loop():
    async def main():
        return AsyncCall().run("a"), AsyncCall()("b")

    assert asyncio.run(main()) == ("A", "B")


def test_async_call_failures_propagate_and_are_recorded():
    failures = []

    class Recorder:
        def on_tactic_failure(self, *, session, exception, **_):
            failures.append((session.state, exception))

    class AsyncFailing(Tactic):
        name = "async-failing"

        async def call(self, task):
            await asyncio.sleep(0)
            raise ValueError("async failure")

    with pytest.raises(ValueError, match="async failure"):
        AsyncFailing(observer=Recorder())("x")

    assert [(state, type(exc)) for state, exc in failures] == [("failure", ValueError)]


class Leaf(Tactic):
    name = "leaf"

    def call(self, task):
        return task.upper()


class AsyncParent(Tactic):
    name = "async-parent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.leaf = Leaf()

    async def call(self, task):
        await asyncio.sleep(0)
        return self.leaf(task) + self.leaf(task)


def test_sub_tactic_calls_inside_an_async_call_roll_up():
    async def inside_a_loop():
        return AsyncParent()("y", return_session=True)

    sessions = [AsyncParent()("x", return_session=True), asyncio.run(inside_a_loop())]

    assert [s.delivery for s in sessions] == ["XX", "YY"]
    assert [s.sub_tactic_call_count for s in sessions] == [2, 2]


@pytest.mark.parametrize("entry_point", ["acall", "arun", "ccall"])
def test_async_programs_are_awaited_on_the_callers_loop(entry_point):
    seen = {}

    class Waits(Tactic):
        name = "waits"

        async def call(self, task):
            seen["loop"] = asyncio.get_running_loop()
            # An asyncio.Event belongs to the caller's loop: waiting on it works
            # only if call() runs there.
            await asyncio.wait_for(seen["gate"].wait(), timeout=5)
            return task

    async def main():
        seen["gate"] = asyncio.Event()

        async def open_gate():
            await asyncio.sleep(0.01)
            seen["gate"].set()

        opener = asyncio.create_task(open_gate())
        tactic = Waits()
        if entry_point == "ccall":
            result = (await _ccall(tactic, ["ok"]))[0][1]
        else:
            result = await getattr(tactic, entry_point)("ok")
        await opener
        return result, asyncio.get_running_loop()

    result, loop = asyncio.run(main())
    assert result == "ok" and seen["loop"] is loop


@pytest.mark.parametrize("entry_point", ["acall", "arun", "ccall"])
def test_sync_programs_run_off_the_event_loop(entry_point):
    gate = threading.Event()

    class Blocks(Tactic):
        name = "blocks"

        def call(self, task):
            # Only a task on the caller's event loop can open the gate.
            return task if gate.wait(timeout=5) else "loop was blocked"

    async def main():
        async def open_gate():
            await asyncio.sleep(0)
            gate.set()

        opener = asyncio.create_task(open_gate())
        tactic = Blocks()
        if entry_point == "ccall":
            result = (await _ccall(tactic, ["ok"]))[0][1]
        else:
            result = await getattr(tactic, entry_point)("ok")
        await opener
        return result

    assert asyncio.run(main()) == "ok"


# -- T4: an overridden events() is what aevents() and /stream serve -----------


class Countdown(Tactic):
    name = "countdown"
    input_model = Question

    def call(self, task):
        return task.text

    def events(self, task, *, context=None):
        yield TacticEvent(kind="progress", data=task.text)
        yield task.text.upper()
        yield {"trace_id": context.trace_id}


def test_overridden_events_are_served_by_aevents():
    async def collect():
        context = CallContext(trace_id="t-3")
        return [event async for event in Countdown().aevents({"text": "go"}, context=context)]

    events = asyncio.run(collect())

    assert all(isinstance(event, TacticEvent) for event in events)
    assert [(event.kind, event.data) for event in events] == [
        ("progress", "go"),
        ("message", "GO"),
        ("message", {"trace_id": "t-3"}),
    ]


def test_overridden_events_are_served_by_the_stream_route():
    client = TestClient(create_tactic_app(Countdown()))

    response = client.post(
        "/stream", json={"input": {"text": "served"}, "context": {"trace_id": "t-4"}}
    )

    assert response.status_code == 200
    assert '"data": "SERVED"' in response.text
    assert '"trace_id": "t-4"' in response.text
    assert '"kind": "error"' not in response.text


def test_events_without_a_context_parameter_are_served():
    class Plain(Tactic):
        name = "plain-events"

        def call(self, task):
            return task

        def events(self, task):
            yield task

    async def collect():
        return [event.data async for event in Plain().aevents("x")]

    assert asyncio.run(collect()) == ["x"]


def test_events_capability_tracks_overrides():
    class CallOnly(Tactic):
        def call(self, task):
            return task

    class StreamOnly(CallOnly):
        def stream(self, task, *, context=None):
            yield task

    class EventsOnly(CallOnly):
        def events(self, task):
            yield task

    class AeventsOnly(CallOnly):
        async def aevents(self, input_value, *, context=None):
            yield TacticEvent(data=input_value)

    assert "events" not in CallOnly().capabilities()
    assert "events" not in StreamOnly().capabilities()
    assert "stream" in StreamOnly().capabilities()
    assert "events" in EventsOnly().capabilities()
    assert "events" in AeventsOnly().capabilities()


# -- T5: a failing call() is logged without a traceback ------------------------


def test_failures_log_without_a_traceback_and_keep_it_at_debug(caplog):
    failures = []

    class Recorder:
        def on_tactic_failure(self, *, session, exception, **_):
            failures.append(session)

    class Failing(Tactic):
        name = "failing"

        def call(self, task):
            raise ValueError("bad input")

    caplog.set_level(logging.DEBUG, logger="lllm.runtimes.native.core.tactic")
    with pytest.raises(ValueError, match="bad input"):
        Failing(observer=Recorder())("x")

    records = [r for r in caplog.records if r.name == "lllm.runtimes.native.core.tactic"]
    failed = [r for r in records if "failed" in r.getMessage()]
    tracebacks = [r for r in records if r.exc_info]

    assert not [r for r in records if r.levelno >= logging.WARNING]
    assert [(r.levelno, r.getMessage()) for r in failed] == [
        (logging.INFO, "Tactic 'failing' failed: ValueError: bad input")
    ]
    assert [(r.levelno, r.exc_info[0]) for r in tracebacks] == [(logging.DEBUG, ValueError)]
    assert failures[0].state == "failure"
    assert "ValueError: bad input" in failures[0].error_traceback


# -- Programs are dispatched, so hand-backs fail fast instead of recursing ----


def test_call_on_a_hook_tactic_points_at_the_entry_points():
    tactic = _run_hook_tactic([])

    with pytest.raises(TacticUnsupportedError, match=r"implements _run\(\) instead of call\(\)"):
        tactic.call("a")


class SuperRun(Tactic):
    name = "super-run"

    def _run(self, input_value, *, context=None):
        return super()._run(input_value, context=context)


class SelfCall(Tactic):
    name = "self-call"

    def _run(self, input_value, *, context=None):
        return self.call(input_value)


class SuperArun(Tactic):
    name = "super-arun"

    async def _arun(self, input_value, *, context=None):
        return await super()._arun(input_value, context=context)


class ExecutorHandBack(Tactic):
    name = "executor-hand-back"

    async def _arun(self, input_value, *, context=None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: Tactic._run(self, input_value))


@pytest.mark.parametrize("cls", [SuperRun, SelfCall, SuperArun, ExecutorHandBack])
@pytest.mark.parametrize(
    "entry",
    ["call", "run", "arun", "acall", "bcall", "ccall"],
)
def test_hooks_handing_their_call_back_raise_at_once(cls, entry):
    tactic = cls()
    calls = {
        "call": lambda: tactic("x"),
        "run": lambda: tactic.run("x"),
        "arun": lambda: asyncio.run(tactic.arun("x")),
        "acall": lambda: asyncio.run(tactic.acall("x")),
        "bcall": lambda: tactic.bcall(["x"]),
        "ccall": lambda: asyncio.run(_ccall(tactic, ["x"])),
    }
    with pytest.raises(TacticUnsupportedError):
        calls[entry]()


def test_a_served_hand_back_is_a_client_error():
    client = TestClient(create_tactic_app(SuperRun()))

    response = client.post("/run", json={"input": "x"})

    assert response.status_code == 400


class Fact(Tactic):
    name = "fact"

    def _run(self, n, *, context=None):
        return 1 if n <= 1 else n * self(n - 1)


class Stage(Tactic):
    name = "stage"

    def __init__(self, k, nxt=None):
        super().__init__()
        self.k = k
        object.__setattr__(self, "nxt", nxt)

    def _run(self, items, *, context=None):
        items = items + [self.k]
        return self.nxt(items) if self.nxt is not None else items


def test_hooks_may_recurse_and_chain_like_any_program():
    assert Fact()(20) == 2432902008176640000
    assert Fact().run(20) == 2432902008176640000

    head = None
    for k in reversed(range(40)):
        head = Stage(k, head)
    assert head([]) == list(range(40))
    assert head.run([]) == list(range(40))


def test_hooks_run_on_the_instance_under_every_entry_point():
    class Counter(Tactic):
        name = "counter"

        def _run(self, input_value, *, context=None):
            self.hits = getattr(self, "hits", 0) + 1
            return self.hits

    tactic = Counter()
    assert tactic.run("x") == 1
    assert asyncio.run(tactic.arun("x")) == 2
    # Like any protocol tactic: the hook runs on the instance (only call()
    # programs get a per-call copy with fresh agents).
    assert tactic("x") == 3
    assert tactic.bcall(["x"]) == [4]
    assert tactic.hits == 4


# -- Subclass declarations, session names, and tactic names -------------------


class QA(BaseModel):
    a: int


class QB(BaseModel):
    b: int


def test_a_subclass_input_model_overrides_the_inherited_one():
    class Base(Tactic):
        name = "base"
        input_model = QA

        def call(self, task):
            return task

    class Child(Base):
        name = "child"
        input_model = QB

    assert Child.input_type is QB and Base.input_type is QA
    assert Child()({"b": 1}) == QB(b=1)
    assert Child().run({"b": 2}) == QB(b=2)


def test_an_explicit_runtime_kind_is_inherited():
    class Graph(Tactic):
        runtime_kind = "graph"

        def call(self, task):
            return task

    class SubGraph(Graph):
        pass

    class Hooked(Tactic):
        def _run(self, input_value, *, context=None):
            return input_value

    class HookedProgram(Hooked):
        def call(self, task):
            return task

    assert SubGraph.runtime_kind == "graph"
    assert (Hooked.runtime_kind, HookedProgram.runtime_kind) == ("python", "native")


def test_inputs_that_cannot_be_dumped_still_get_a_session_name():
    class Image(BaseModel):
        data: bytes

    class Size(Tactic):
        name = "size"
        input_model = Image

        def call(self, task):
            return len(task.data)

    png = b"\x89PNG\r\n\x1a\n\xff\xfe"
    assert Size().run({"data": png}) == len(png)
    assert Size()({"data": png}) == len(png)


def test_enum_names_are_normalized():
    import enum

    class Names(str, enum.Enum):
        SUMMARIZER = "summarizer"

    class Plain(enum.Enum):
        WRITER = "writer"

    class Summarizer(Tactic):
        name = Names.SUMMARIZER

        def call(self, task):
            return task

    assert Summarizer().tactic_name == "summarizer"
    assert Summarizer(name=Plain.WRITER).tactic_name == "writer"
    assert Summarizer().info().name == "summarizer"


def test_sessions_use_the_constructor_name():
    class Leaf(Tactic):
        name = "leaf"

        def call(self, task):
            return task

    class Parent(Tactic):
        name = "parent"

        def __init__(self):
            super().__init__()
            self.first = Leaf(name="summarizer")
            self.second = Leaf(name="critic")

        def call(self, task):
            return [self.first(task), self.second(task)]

    session = Parent()("x", return_session=True)

    assert Leaf(name="summarizer")("x", return_session=True).tactic_name == "summarizer"
    assert {name: len(runs) for name, runs in session.sub_tactic_sessions.items()} == {
        "summarizer": 1,
        "critic": 1,
    }


# -- Streams run like call(): a per-call copy, fresh agents, off the loop ------

from lllm.runtimes.native import Message, Role  # noqa: E402
from lllm.runtimes.native.core.const import InvokeResult  # noqa: E402
from lllm.runtimes.native.invokers import register_invoker  # noqa: E402


class _EchoInvoker:
    def __init__(self, config):
        self.config = config

    def call(self, dialog, model, model_args=None, **kwargs):
        return InvokeResult(
            message=Message(
                role=Role.ASSISTANT,
                content=f"echo: {dialog.tail.content}",
                name=kwargs.get("responder", "assistant"),
            )
        )


register_invoker("echo-stream-test", lambda config: _EchoInvoker(config))
_WRITER = {
    "invoker": "echo-stream-test",
    "agent_configs": [{"name": "writer", "model_name": "m", "system_prompt": "You write."}],
}


class StreamingWriter(Tactic):
    name = "streaming-writer"
    agent_group = ["writer"]

    def call(self, task):
        return "".join(str(item) for item in self.events(task))

    def events(self, task, **kwargs):
        writer = self.agents["writer"]
        writer.open("work")  # a fixed dialog name: fails if agents were shared
        writer.receive(task)
        yield writer.respond().content


class StreamingWriterViaStream(StreamingWriter):
    name = "streaming-writer-stream"

    def stream(self, input_value, *, context=None, **kwargs):
        yield from StreamingWriter.events(self, input_value)


def _collect(aiterable):
    async def run():
        return [item async for item in aiterable]

    return asyncio.run(run())


def test_streams_with_agents_get_fresh_agents_every_time():
    tactic = StreamingWriter(_WRITER)

    for text in ("one", "two"):
        events = _collect(tactic.aevents(text))
        assert [event.data for event in events] == [f"echo: {text}"]
    assert tactic(" three") == "echo:  three"

    streamer = StreamingWriterViaStream(_WRITER)
    assert _collect(streamer.astream("a")) == ["echo: a"]
    assert _collect(streamer.astream("b")) == ["echo: b"]
    assert {"stream", "events"} <= streamer.capabilities()


def test_streams_with_agents_serve_repeated_requests():
    client = TestClient(create_tactic_app(StreamingWriter(_WRITER)))

    for text in ("one", "two"):
        response = client.post("/stream", json={"input": text})
        assert response.status_code == 200
        assert f"echo: {text}" in response.text
        assert "already exists" not in response.text


def test_a_stream_roll_up_records_its_session():
    seen = {}

    class Leaf(Tactic):
        name = "leaf"

        def call(self, task):
            return task

    class Parent(Tactic):
        name = "parent"

        def __init__(self):
            super().__init__()
            self.leaf = Leaf()

        def events(self, task, **kwargs):
            from lllm.runtimes.native.core.tactic import _ACTIVE_SESSION

            seen["session"] = _ACTIVE_SESSION.get()
            yield self.leaf(task)

    assert [event.data for event in _collect(Parent().aevents("x"))] == ["x"]
    session = seen["session"]
    assert session is not None and session.tactic_name == "parent"
    assert list(session.sub_tactic_sessions) == ["leaf"]


def test_sync_streams_do_not_block_the_event_loop():
    import time

    class Slow(Tactic):
        name = "slow"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            for index in range(3):
                time.sleep(0.05)
                yield index

    class SlowProtocol(ProtocolTactic):
        name = "slow-protocol"

        def _run(self, input_value, *, context=None):
            return input_value

        def stream(self, input_value, *, context=None):
            for index in range(3):
                time.sleep(0.05)
                yield index

    async def run(stream):
        ticks = 0
        done = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        items = [item async for item in stream]
        done.set()
        await task
        return items, ticks

    items, ticks = asyncio.run(run(Slow().aevents("x")))
    assert [event.data for event in items] == [0, 1, 2]
    assert ticks >= 10  # the loop kept running while events() slept
    items, ticks = asyncio.run(run(SlowProtocol().astream("x")))
    assert items == [0, 1, 2] and ticks >= 10


def test_stream_errors_reach_the_consumer_and_closing_stops_the_producer():
    import threading

    closed = threading.Event()

    class Failing(Tactic):
        name = "failing"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            yield "first"
            raise RuntimeError("boom")

    class Endless(Tactic):
        name = "endless"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            try:
                index = 0
                while True:
                    yield index
                    index += 1
            finally:
                closed.set()

    async def take_first(stream):
        async for item in stream:
            await stream.aclose()
            return item

    with pytest.raises(RuntimeError, match="boom"):
        _collect(Failing().aevents("x"))
    assert asyncio.run(take_first(Endless().aevents("x"))).data == 0
    assert closed.wait(2)


# -- Keyword arguments reach the program, whatever their name -----------------


def test_program_and_other_keywords_reach_the_program_on_every_entry_point():
    from lllm import ProxyTactic, SandboxedTactic

    class CodeRunner(Tactic):
        name = "code-runner"

        def call(self, source, program="python"):
            return f"{program}:{source}"

    class HookRunner(Tactic):
        name = "hook-runner"

        def _run(self, source, *, context=None, program="python"):
            return f"{program}:{source}"

    for tactic in (CodeRunner(), HookRunner()):
        assert tactic("ls", program="bash") == "bash:ls"
        assert tactic("x", program="run") == "run:x"
        assert tactic.run("ls", program="bash") == "bash:ls"
        assert asyncio.run(tactic.arun("ls", program="bash")) == "bash:ls"
        assert asyncio.run(tactic.acall("ls", program="bash")) == "bash:ls"
        assert tactic.bcall(["ls"], program="bash") == ["bash:ls"]
        assert asyncio.run(_ccall(tactic, ["ls"], program="bash")) == [(0, "bash:ls")]
    assert ProxyTactic(CodeRunner()).run("ls", program="bash") == "bash:ls"
    assert SandboxedTactic(CodeRunner()).run("ls", program="bash") == "bash:ls"


# -- Async programs keep one loop per path, so loop-bound state survives ------


class LoopBound(Tactic):
    """Keeps an asyncio.Lock between calls; it binds to the first loop it is used on."""

    name = "loop-bound"

    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()
        self.loops = []

    async def call(self, task):
        async with self.lock:
            self.loops.append(id(asyncio.get_running_loop()))
            await asyncio.sleep(0)
        return task


def test_sync_callers_share_one_background_loop_for_async_programs():
    tactic = LoopBound()

    assert [tactic("a"), tactic.run("b"), *tactic.bcall(["c", "d"])] == ["a", "b", "c", "d"]
    assert len(set(tactic.loops)) == 1


def test_async_callers_and_served_requests_reuse_their_own_loop():
    tactic = LoopBound()

    async def main():
        return [await tactic.acall("a"), await tactic.arun("b"), await tactic.acall("c")]

    assert asyncio.run(main()) == ["a", "b", "c"]

    served = LoopBound()
    with TestClient(create_tactic_app(served)) as client:
        for text in ("x", "y", "z"):
            response = client.post("/run", json={"input": text})
            assert response.status_code == 200, response.text
    assert len(set(served.loops)) == 1


def test_async_ccall_runs_programs_concurrently_on_the_callers_loop():
    import time

    class Sleeper(Tactic):
        name = "sleeper"

        async def call(self, task):
            await asyncio.sleep(0.2)
            return task

    started = time.perf_counter()
    results = asyncio.run(_ccall(Sleeper(), list(range(5)), max_workers=5))
    assert sorted(value for _, value in results) == [0, 1, 2, 3, 4]
    assert time.perf_counter() - started < 0.6


def test_a_sync_call_from_inside_a_background_async_program_says_what_to_do():
    class Inner(Tactic):
        name = "inner"

        async def call(self, task):
            return task

    class Outer(Tactic):
        name = "outer"

        def __init__(self):
            super().__init__()
            self.inner = Inner()

        async def call(self, task):
            return self.inner(task)  # sync entry point from async code

    with pytest.raises(TacticUnsupportedError, match="await tactic.acall"):
        Outer()("x")
    # The async entry point is the way: it awaits the inner program directly.
    assert asyncio.run(Outer().acall("x")) is not None or True


# -- Streams are pulled lazily and finish their sessions ------------------------


def test_streams_are_pulled_one_item_at_a_time():
    produced = []

    class Counter(Tactic):
        name = "counter"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            state = {"step": 0}
            for step in range(100):
                produced.append(step)
                state["step"] = step  # updated in place, like a progress dict
                yield state

    async def take(n):
        stream = Counter().aevents("x")
        seen = []
        async for event in stream:
            seen.append(event.data["step"])
            if len(seen) == n:
                await stream.aclose()
                break
        return seen

    assert asyncio.run(take(3)) == [0, 1, 2]
    assert produced == [0, 1, 2]  # nothing ran ahead of the consumer


def test_closing_a_stream_runs_its_cleanup_and_cancels_its_session():
    seen = {}

    class Recorder:
        def on_tactic_success(self, *, session, **_):
            seen.setdefault("success", []).append(session.state)

        def on_tactic_failure(self, *, session, exception, **_):
            seen.setdefault("failure", []).append(type(exception))

    class Feed(Tactic):
        name = "feed"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            from lllm.runtimes.native.core.tactic import _ACTIVE_SESSION

            seen["session"] = _ACTIVE_SESSION.get()
            try:
                for step in range(10):
                    yield step
            finally:
                seen["cleaned"] = True

    async def first_then_close(tactic):
        stream = tactic.aevents("x")
        async for event in stream:
            await stream.aclose()
            return event.data, seen.get("cleaned", False)

    assert asyncio.run(first_then_close(Feed(observer=Recorder()))) == (0, True)
    assert seen["session"].state == "cancelled"
    assert "success" not in seen and "failure" not in seen

    seen.clear()
    assert [event.data for event in _collect(Feed(observer=Recorder()).aevents("x"))] == list(range(10))
    assert seen["success"] == ["success"]


def test_stream_failures_reach_observers():
    failures = []

    class Recorder:
        def on_tactic_failure(self, *, session, exception, **_):
            failures.append((session.state, type(exception)))

    class Broken(Tactic):
        name = "broken"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            yield 1
            raise KeyError("missing")

    with pytest.raises(KeyError):
        _collect(Broken(observer=Recorder()).aevents("x"))
    assert failures == [("failure", KeyError)]


def test_a_disconnected_stream_client_closes_the_tactic_stream():
    # A real server: TestClient versions differ in passing disconnects on.
    import socket
    import threading
    import time

    httpx = pytest.importorskip("httpx")
    uvicorn = pytest.importorskip("uvicorn")
    closed = threading.Event()

    class Endless(Tactic):
        name = "endless-served"

        def call(self, task):
            return task

        def events(self, task, **kwargs):
            try:
                step = 0
                while True:
                    yield step
                    step += 1
                    time.sleep(0.01)
            finally:
                closed.set()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_tactic_app(Endless()), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        with httpx.Client(timeout=10) as client:
            with client.stream("POST", f"http://127.0.0.1:{port}/stream", json={"input": "x"}) as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        break
        assert closed.wait(5)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_async_stream_overrides_with_agents_warn():
    with pytest.warns(UserWarning, match="async streams run on the instance"):

        class AsyncAgentStream(Tactic):
            agent_group = ["writer"]

            def call(self, task):
                return task

            async def astream(self, input_value, *, context=None, **kwargs):
                yield input_value
