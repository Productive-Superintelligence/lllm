"""lllm.testing answers native agents' model calls without a provider."""

import pytest

pytest.importorskip("litellm", reason="lllm.testing drives the native runtime")

from lllm import Prompt, Tactic, tool  # noqa: E402
from lllm.testing import FunctionModel, ScriptedModel, ToolCall  # noqa: E402


def test_scripted_replies_come_in_order_and_the_last_repeats():
    agent = Tactic.quick(system_prompt="Be terse.", model="gpt-4o-mini")
    agent.open("chat")
    replies = []
    with ScriptedModel(["one", "two"]):
        for text in ("a", "b", "c"):
            agent.receive(text)
            replies.append(agent.respond().content)
    assert replies == ["one", "two", "two"]


@pytest.mark.parametrize("model", ["gpt-4o-mini", "anthropic/claude-sonnet-4-5"])
def test_requests_show_the_system_prompt_for_every_provider(model):
    # OpenAI requests carry the system prompt with the "developer" role.
    with FunctionModel(lambda request: "ok") as fake:
        Tactic.quick("Hello", system_prompt="Be terse.", model=model)
    (request,) = fake.requests
    assert request.system == "Be terse."
    assert request.text == "Hello"


def test_tool_call_replies_run_the_real_tool():
    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    ask = Prompt(path="math/add", prompt="What is {a} + {b}?", function_list=[add])
    agent = Tactic.quick(system_prompt="You add numbers.", model="gpt-4o-mini")
    agent.open("math")
    agent.receive_prompt(ask, {"a": 2, "b": 3})
    with ScriptedModel([ToolCall("add", {"a": 2, "b": 3}), "2 + 3 = 5"]) as fake:
        reply = agent.respond()
    assert reply.content == "2 + 3 = 5"
    assert fake.requests[0].tools == ["add"]
    assert "5" in fake.requests[1].tool_results()[0]


def test_tool_calls_work_when_the_agent_streams():
    from lllm.runtimes.native.invokers.base import BaseStreamHandler

    class Collect(BaseStreamHandler):
        def __init__(self):
            self.pieces = []

        def handle_chunk(self, chunk_content, chunk_response):
            self.pieces.append(chunk_content)

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    ask = Prompt(path="math/add", prompt="What is {a} + {b}?", function_list=[add])
    agent = Tactic.quick(system_prompt="You add numbers.", model="gpt-4o-mini")
    agent.stream_handler = Collect()
    agent.open("math")
    agent.receive_prompt(ask, {"a": 2, "b": 3})
    script = [[ToolCall("add", {"a": 2, "b": 3}), ToolCall("add", {"a": 1, "b": 1})], "5 and 2"]
    with ScriptedModel(script):
        session = agent.respond(return_session=True)
    assert session.delivery.content == "5 and 2"
    assert [call.result for call in session.tool_calls] == [5, 2]
    assert "".join(agent.stream_handler.pieces) == "5 and 2"


def test_an_inner_fake_model_takes_over_and_restores_the_outer_one():
    with FunctionModel(lambda request: "outer"):
        with ScriptedModel(["inner"]):
            assert Tactic.quick("Hi", model="gpt-4o-mini").content == "inner"
        assert Tactic.quick("Hi", model="gpt-4o-mini").content == "outer"


def test_a_scripted_model_needs_a_reply():
    with pytest.raises(ValueError):
        ScriptedModel([])
