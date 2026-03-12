import uuid

import pytest

from lllm.core.const import Roles, APITypes
from lllm.core.models import Function, FunctionCall, Message, Prompt
from lllm.invokers.base import BaseInvoker
from tests.helpers.agent_utils import make_agent


class FakeInvoker(BaseInvoker):
    """Invoker that returns preseeded Message objects for each call."""

    def __init__(self, responses):
        self._responses = list(responses)

    def call(
        self,
        dialog,
        prompt,
        model,
        model_args=None,
        parser_args=None,
        responder='assistant',
        extra=None,
        api_type=APITypes.COMPLETION,
    ):
        if not self._responses:
            raise AssertionError("FakeInvoker received more calls than responses")
        response = self._responses.pop(0)
        return response

    def stream(self, *args, **kwargs):
        raise NotImplementedError


def test_agent_call_returns_message_without_tools(log_config):
    system_prompt = Prompt(path="tests/system", prompt="You are a tester.")
    query_prompt = Prompt(path="tests/query", prompt="User task: {task}")

    invoker = FakeInvoker(
        [
            Message(
                role=Roles.ASSISTANT,
                name="assistant",
                content="All done!",
                model="gpt-4o-mini",
            )
        ]
    )

    agent = make_agent(system_prompt, invoker, log_config)
    dialog = agent.init_dialog()
    dialog.send_message(query_prompt, {"task": "demo"})

    response, dialog, interrupts = agent.call(dialog)
    assert response.content == "All done!"
    assert interrupts == []
    assert dialog.tail == response


def test_agent_call_executes_registered_function(log_config):
    calls = []

    def _echo(value: str) -> str:
        calls.append(value)
        return f"echo:{value}"

    tool = Function(
        name="echo",
        description="Echo input",
        properties={"value": {"type": "string"}},
        required=["value"],
    )
    tool.link_function(_echo)

    system_prompt = Prompt(path="tests/tool/system", prompt="Use tools when needed.")
    task_prompt = Prompt(path="tests/tool/query", prompt="Task: {task}", function_list=[tool])

    tool_call_message = Message(
        role=Roles.TOOL_CALL,
        name="assistant",
        content="Calling echo",
        function_calls=[
            FunctionCall(id=str(uuid.uuid4()), name="echo", arguments={"value": "foo"})
        ],
        model="gpt-4o-mini",
    )
    final_message = Message(
        role=Roles.ASSISTANT,
        name="assistant",
        content="Tool results processed.",
        model="gpt-4o-mini",
    )

    invoker = FakeInvoker([tool_call_message, final_message])
    agent = make_agent(system_prompt, invoker, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"task": "use the tool"})

    response, dialog, interrupts = agent.call(dialog)

    assert response.content == "Tool results processed."
    assert len(interrupts) == 1
    assert calls == ["foo"]
    assert interrupts[0].result == "echo:foo"


def test_agent_call_uses_tool_role_for_response_api(log_config):
    def _noop_tool(value: str) -> str:
        return value.upper()

    tool = Function(
        name="shout",
        description="Upper-case text",
        properties={"value": {"type": "string"}},
        required=["value"],
    )
    tool.link_function(_noop_tool)

    system_prompt = Prompt(path="tests/response/system", prompt="Use tools.")
    task_prompt = Prompt(
        path="tests/response/query",
        prompt="Task: {task}",
        function_list=[tool],
    )

    tool_call_message = Message(
        role=Roles.TOOL_CALL,
        name="assistant",
        content="Calling shout",
        function_calls=[
            FunctionCall(id="call_123", name="shout", arguments={"value": "ping"})
        ],
        model="gpt-4o-mini",
        api_type=APITypes.RESPONSE,
    )
    final_message = Message(
        role=Roles.ASSISTANT,
        name="assistant",
        content="Done.",
        model="gpt-4o-mini",
        api_type=APITypes.RESPONSE,
    )

    invoker = FakeInvoker([tool_call_message, final_message])
    agent = make_agent(system_prompt, invoker, log_config)
    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"task": "run shout"})

    agent.call(dialog)

    tool_messages = [msg for msg in dialog.messages if msg.role == Roles.USER]
    assert tool_messages, "response-api tool results should surface as USER role"
    assert all(msg.role == Roles.USER for msg in tool_messages)
