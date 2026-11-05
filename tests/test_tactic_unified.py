"""One Tactic: the native program model on top of the typed protocol."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import lllm
from lllm import CallContext, Tactic, TacticUnsupportedError
from lllm.protocol import Tactic as ProtocolTactic
from lllm.runtimes.native import Message, NativeTactic, Role
from lllm.runtimes.native.core.const import InvokeResult
from lllm.runtimes.native.invokers import register_invoker
from lllm.services import create_tactic_app


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str
    trace_id: str | None = None


class Shout(Tactic):
    """Shout the question back."""

    name = "shout"
    input_model = Question
    output_model = Answer

    def call(self, task, *, context=None):
        return Answer(text=task.text.upper(), trace_id=getattr(context, "trace_id", None))


class EchoInvoker:
    def __init__(self, config):
        self.config = config

    def call(self, dialog, model, model_args=None, **kwargs):
        return InvokeResult(
            message=Message(
                role=Role.ASSISTANT,
                content=f"{model}: {dialog.tail.content}",
                name=kwargs.get("responder", "assistant"),
            )
        )


register_invoker("echo-test", lambda config: EchoInvoker(config))

AGENTS = {
    "invoker": "echo-test",
    "agent_configs": [
        {"name": "writer", "model_name": "model-a", "system_prompt": "You write."},
        {"name": "critic", "model_name": "model-b", "system_prompt": "You critique."},
    ],
}


class Pipeline(Tactic):
    name = "pipeline"
    agent_group = ["writer", "critic"]

    def call(self, task: str) -> str:
        writer, critic = self.agents["writer"], self.agents["critic"]
        writer.open("draft")
        writer.receive(task)
        draft = writer.respond().content
        critic.open("review")
        critic.receive(draft)
        return critic.respond().content


def test_lllm_tactic_is_the_program_and_a_protocol_tactic():
    assert lllm.Tactic is Tactic
    assert issubclass(Tactic, ProtocolTactic)
    assert callable(Tactic.quick)


def test_call_style_tactics_validate_and_publish_their_contract():
    tactic = Shout()

    assert tactic({"text": "hi"}) == Answer(text="HI")
    assert tactic.run({"text": "hi"}).text == "HI"
    info = tactic.info()
    assert info.runtime_kind == "native"
    assert info.description == "Shout the question back."
    assert info.input_schema["properties"] == {"text": {"title": "Text", "type": "string"}}
    assert {"run", "arun", "call", "acall"} <= set(info.capabilities)
    with pytest.raises(Exception):
        tactic.run({"wrong": "field"})


def test_call_context_reaches_call_and_sessions_record_the_run():
    tactic = Shout()

    output = tactic.run({"text": "hi"}, context=CallContext(trace_id="t-1"))
    session = tactic({"text": "hi"}, return_session=True)

    assert output.trace_id == "t-1"
    assert session.state == "success"
    assert session.tactic_name == "shout"


def test_call_style_tactics_serve_without_an_adapter():
    client = TestClient(create_tactic_app(Shout()))

    response = client.post(
        "/run", json={"input": {"text": "served"}, "context": {"trace_id": "t-2"}}
    )

    assert response.status_code == 200
    assert response.json()["output"] == {"text": "SERVED", "trace_id": "t-2"}
    assert client.get("/info").json()["runtime_kind"] == "native"


def test_protocol_style_subclasses_still_work_and_report_python():
    class Upper(Tactic[Question, Answer]):
        name = "upper"
        input_type = Question
        output_type = Answer

        def _run(self, input_value, *, context=None):
            return Answer(text=input_value.text.upper())

    assert Upper().run({"text": "ok"}).text == "OK"
    assert Upper().info().runtime_kind == "python"
    assert Upper(name="renamed").info().name == "renamed"


def test_agent_group_builds_fresh_agents_per_call_and_tracks_them():
    tactic = Pipeline(AGENTS)

    session = tactic("draft a haiku", return_session=True)

    assert session.delivery == "model-b: model-a: draft a haiku"
    assert session.agent_call_count == 2
    assert sorted(tactic.agents) == ["critic", "writer"]
    assert tactic("again") == "model-b: model-a: again"
    assert asyncio.run(tactic.arun("async")) == "model-b: model-a: async"


def test_batch_calls_run_the_program_concurrently():
    assert Pipeline(AGENTS).bcall(["a", "b"]) == ["model-b: model-a: a", "model-b: model-a: b"]


def test_tactics_without_a_program_say_so():
    class Empty(Tactic):
        name = "empty"

    with pytest.raises(TacticUnsupportedError, match="call\\(\\) or _run\\(\\)"):
        Empty().run("x")


def test_native_tactic_still_requires_an_agent_group():
    class NoAgents(NativeTactic):
        name = "no-agents"

        def call(self, task):
            return task

    with pytest.raises(AssertionError, match="agent_group"):
        NoAgents()
    assert issubclass(NativeTactic, Tactic)


class Leaf(Tactic):
    name = "leaf"

    def call(self, task):
        if task == "fail":
            raise ValueError("leaf failed")
        return task.upper()


class Parent(Tactic):
    name = "parent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.leaf = Leaf()

    def call(self, task):
        if task == "batch":
            return "+".join(self.leaf.bcall(["a", "b"]))
        if task == "fail":
            try:
                self.leaf("fail")
            except ValueError:
                return "handled"
        return self.leaf(task) + self.leaf(task)


def test_sub_tactic_calls_roll_up_into_the_parent_session():
    session = Parent()("x", return_session=True)

    assert session.delivery == "XX"
    assert session.sub_tactic_call_count == 2
    assert [s.state for s in session.sub_tactic_sessions["leaf"]] == ["success", "success"]


def test_sub_tactic_sessions_link_across_batch_threads_and_failures():
    batch = Parent()("batch", return_session=True)
    failed = Parent()("fail", return_session=True)

    assert batch.delivery == "A+B"
    assert batch.sub_tactic_call_count == 2
    assert [s.state for s in failed.sub_tactic_sessions["leaf"]] == ["failure"]


def test_top_level_calls_do_not_link_to_each_other():
    leaf = Leaf()
    leaf("a")

    assert leaf("b", return_session=True).sub_tactic_call_count == 0


def test_the_native_vocabulary_is_importable_from_lllm():
    from lllm import (
        Agent,
        BaseProxy,
        Dialog,
        Prompt,
        build_tactic,
        install_package,
        load_runtime,
        resolve_config,
        tool,
    )
    from lllm.runtimes import native

    assert Prompt is native.Prompt and Agent is native.Agent and Dialog is native.Dialog
    assert all(callable(fn) for fn in (build_tactic, install_package, load_runtime, resolve_config, tool))
    assert BaseProxy.__name__ == "BaseProxy"


def test_missing_resources_point_to_load_runtime():
    from lllm import Runtime

    with pytest.raises(KeyError, match="load_runtime"):
        Runtime().get("writer/system", resource_type="prompt")
