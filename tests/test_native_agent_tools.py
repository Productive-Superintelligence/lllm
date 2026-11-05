"""Agent-level tools stay available on every turn; rate limits are bounded."""

import pytest

from lllm.runtimes.native import Agent, FunctionCall, Message, Prompt, Role, tool
from lllm.runtimes.native.core.const import InvokeResult


@tool(description="Look up the weather for a city")
def weather(city: str) -> str:
    return f"sunny in {city}"


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


class ScriptedInvoker:
    """Replays scripted replies and records the tools offered on each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.offered = []

    def call(self, dialog, model, model_args=None, **kwargs):
        self.offered.append(sorted(dialog.top_prompt.functions))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return InvokeResult(message=reply)


def text(content):
    return Message(role=Role.ASSISTANT, content=content, name="assistant")


def calls(*function_calls):
    return Message(
        role=Role.ASSISTANT,
        content="",
        name="assistant",
        function_calls=list(function_calls),
    )


def make_agent(invoker, **fields):
    return Agent(
        name="assistant",
        system_prompt=Prompt(
            path="tools/system", prompt="You help.", function_list=[weather]
        ),
        model="fake",
        llm_invoker=invoker,
        **fields,
    )


def test_system_prompt_tools_survive_a_plain_receive():
    invoker = ScriptedInvoker(text("hi"), text("again"))
    agent = make_agent(invoker)
    agent.open("chat")

    agent.receive("hello")
    agent.respond()
    agent.receive("what about now?")
    agent.respond()

    assert invoker.offered == [["weather"], ["weather"]]


def test_agent_tools_run_after_receive_and_through_the_tool_loop():
    invoker = ScriptedInvoker(
        calls(FunctionCall(id="c1", name="weather", arguments={"city": "Paris"})),
        text("It is sunny in Paris."),
    )
    agent = make_agent(invoker)
    agent.open("chat")
    agent.receive("Weather in Paris?")

    reply = agent.respond()

    assert reply.content == "It is sunny in Paris."
    tool_messages = [m for m in agent.current_dialog.messages if m.role == Role.TOOL]
    assert "sunny in Paris" in tool_messages[0].content
    assert invoker.offered == [["weather"], ["weather"]]


def test_turn_prompts_add_tools_and_win_name_clashes():
    invoker = ScriptedInvoker(text("ok"))
    agent = make_agent(invoker)
    agent.open("chat")
    turn = Prompt(path="tools/turn", prompt="Compute {x}.", function_list=[add])

    agent.receive_prompt(turn, {"x": "1+1"})
    agent.respond()

    assert invoker.offered == [["add", "weather"]]


def test_a_turn_can_opt_out_of_agent_tools():
    invoker = ScriptedInvoker(text("final"))
    agent = make_agent(invoker)
    agent.open("chat")
    final = Prompt(path="tools/final", prompt="Answer now.", agent_tools=False)

    agent.receive_prompt(final)
    agent.respond()

    assert invoker.offered == [[]]


class RateLimitError(Exception):
    status_code = 429


def test_rate_limits_back_off_and_then_give_up(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lllm.runtimes.native.core.agent.time.sleep", sleeps.append)
    invoker = ScriptedInvoker(RateLimitError("slow down"), RateLimitError("slow down"))
    agent = make_agent(invoker, max_rate_limit_retries=1)
    agent.open("chat")
    agent.receive("hello")

    with pytest.raises(RateLimitError):
        agent.respond()

    assert len(sleeps) == 1 and 2.0 <= sleeps[0] < 3.0


def test_rate_limits_recover_when_the_provider_does(monkeypatch):
    monkeypatch.setattr("lllm.runtimes.native.core.agent.time.sleep", lambda s: None)
    invoker = ScriptedInvoker(RateLimitError("slow down"), text("recovered"))
    agent = make_agent(invoker)
    agent.open("chat")
    agent.receive("hello")

    assert agent.respond().content == "recovered"
