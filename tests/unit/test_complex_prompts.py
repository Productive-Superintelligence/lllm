import json
import textwrap
from types import SimpleNamespace

import pytest

from lllm.core.const import APITypes, ParseError, Roles
from lllm.core.models import Function, FunctionCall, Message, Prompt
from lllm.invokers.litellm import LiteLLMInvoker
from tests.helpers.agent_utils import make_agent
from tests.helpers.scripted_invoker import ScriptedInvoker


def _proposition_tree_parser(message: str, current_nodes: list[str]):
    """Parser inspired by the Analytica analyzer prompts."""
    if "```json" not in message:
        raise ParseError("Please provide one and only one JSON block")
    block = message.split("```json", 1)[1]
    if "```" not in block:
        raise ParseError("JSON block must be fenced with ```json ... ```")
    json_payload = block.split("```", 1)[0]
    try:
        data = json.loads(json_payload.strip())
    except Exception as exc:
        raise ParseError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ParseError("JSON payload must be a non-empty list of proposition edges")

    seen_ids = set()
    nodes = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise ParseError("Each entry must be an object")
        parent = entry.get("parent")
        children = entry.get("children")
        causality = entry.get("causality")
        if not isinstance(parent, str):
            raise ParseError("parent must be a string")
        if parent not in current_nodes:
            raise ParseError(f"Root node {parent} is not part of the current tree")
        if not isinstance(children, dict) or not children:
            raise ParseError("children must be a non-empty object")
        if not isinstance(causality, str):
            raise ParseError("causality must be provided")
        for child_id, statement in children.items():
            if child_id in seen_ids:
                raise ParseError(f"Duplicate child id detected: {child_id}")
            seen_ids.add(child_id)
            if not statement.strip():
                raise ParseError("Child proposition statements must be non-empty")
        nodes[parent] = {"children": children, "causality": causality}

    return {"root": data[0]["parent"], "nodes": nodes}


def test_agent_recovers_from_parser_error_and_uses_exception_prompt(log_config):
    system_prompt = Prompt(path="complex/system", prompt="You are a planner.")
    analyze_prompt = Prompt(
        path="complex/analyze",
        prompt="Analyze: {topic}",
        parser=_proposition_tree_parser,
        md_tags=["json"],
        required_md_tags=["json"],
    )

    bad_payload = "No json block here."
    good_payload = textwrap.dedent(
        """
        Thorough analysis.
        ```json
        [
            {
                "parent": "P0",
                "children": {
                    "P1": "Growth depends on policy support",
                    "P2": "Funding cycles remain liquid"
                },
                "causality": "Both drivers must hold for P0 to be true."
            }
        ]
        ```
        """
    ).strip()

    invoker = ScriptedInvoker(
        [
            {"content": bad_payload},
            {"content": good_payload},
        ]
    )

    agent = make_agent(system_prompt, invoker, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(analyze_prompt, {"topic": "Emerging market resilience"}, role=Roles.USER)

    response, dialog, interrupts = agent.call(dialog, parser_args={"current_nodes": ["P0"]})

    assert invoker.call_count == 2, "Agent should retry once after parser failure"
    assert invoker.errors[0], "First call should record a parser error"
    assert "one and only one JSON block" in str(invoker.errors[0][0])
    assert response.parsed["root"] == "P0"
    assert response.parsed["nodes"]["P0"]["children"]["P1"].startswith("Growth")
    assert interrupts == []


def test_agent_surfaces_duplicate_tool_call_warning(log_config):
    calls = []

    def _echo(value: str) -> str:
        calls.append(value)
        return f"echo:{value}"

    tool = Function(
        name="echo",
        description="Echo text back.",
        properties={"value": {"type": "string"}},
        required=["value"],
    )
    tool.link_function(_echo)

    system_prompt = Prompt(path="complex/tool/system", prompt="Use tools carefully.")
    task_prompt = Prompt(
        path="complex/tool/query",
        prompt="Run the echo tool for {value}.",
        function_list=[tool],
        interrupt_prompt="Result: {call_results}",
        interrupt_final_prompt="All tools done. Summarize.",
    )

    scripts = [
        {
            "role": Roles.TOOL_CALL,
            "content": "Calling echo",
            "function_calls": [
                FunctionCall(id="call-1", name="echo", arguments={"value": "alpha"})
            ],
        },
        {
            "role": Roles.TOOL_CALL,
            "content": "Calling echo again",
            "function_calls": [
                FunctionCall(id="call-2", name="echo", arguments={"value": "alpha"})
            ],
        },
        {"role": Roles.ASSISTANT, "content": "All done."},
    ]

    invoker = ScriptedInvoker(scripts)
    agent = make_agent(system_prompt, invoker, log_config)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"value": "alpha"})

    response, dialog, interrupts = agent.call(dialog)

    assert invoker.call_count == 3
    assert response.content == "All done."
    assert calls == ["alpha"]
    assert len(interrupts) == 1, "Duplicate tool calls should not append new interrupts"

    duplicate_warnings = [
        msg.content for msg in dialog.messages if "has been called earlier" in msg.content
    ]
    assert duplicate_warnings, "Dialog should capture duplicate call warnings"


def test_response_api_tool_results_emit_user_role(log_config):
    calls = []

    def _lookup(symbol: str) -> str:
        calls.append(symbol)
        return f"quote:{symbol}"

    tool = Function(
        name="lookup_quote",
        description="Fetch a quote.",
        properties={"symbol": {"type": "string"}},
        required=["symbol"],
    )
    tool.link_function(_lookup)

    system_prompt = Prompt(path="complex/response/system", prompt="Use response api.")
    task_prompt = Prompt(
        path="complex/response/query",
        prompt="Get info on {symbol}",
        function_list=[tool],
        interrupt_prompt="Tool output: {call_results}",
        interrupt_final_prompt="Finish up.",
    )

    scripts = [
        {
            "role": Roles.TOOL_CALL,
            "content": "Need tool result",
            "function_calls": [
                FunctionCall(id="resp-call-1", name="lookup_quote", arguments={"symbol": "XYZ"})
            ],
            "api_type": APITypes.RESPONSE,
        },
        {
            "role": Roles.ASSISTANT,
            "content": "Here is the summary.",
            "api_type": APITypes.RESPONSE,
        },
    ]

    invoker = ScriptedInvoker(scripts)
    agent = make_agent(system_prompt, invoker, log_config, api_type=APITypes.RESPONSE)

    dialog = agent.init_dialog()
    dialog.send_message(task_prompt, {"symbol": "XYZ"})

    response, dialog, interrupts = agent.call(dialog)

    assert calls == ["XYZ"]
    assert response.content == "Here is the summary."
    assert interrupts and interrupts[0].name == "lookup_quote"

    tool_role_messages = [
        msg for msg in dialog.messages if msg.role == Roles.USER and msg.name == "function"
    ]
    assert tool_role_messages, "Response API tool replies should use user-role messages"


def test_response_api_includes_search_and_computer_tools(monkeypatch):
    class StubCard: 
        def __init__(self):
            self.features = ["web_search", "computer_use"]
            self.base_url = None

    class FakeUsage:
        def __init__(self, payload):
            self._payload = payload

        def model_dump_json(self):
            return json.dumps(self._payload)

    class FakeResponse:
        def __init__(self):
            self.output_text = "Final search-backed answer."
            self.output = [SimpleNamespace(type="output_text", text=self.output_text)]
            self.reasoning = SimpleNamespace(model_dump_json=lambda: json.dumps({"steps": []}))
            self.usage = FakeUsage(
                {"prompt_tokens": 10, "completion_tokens": 5, "cached_prompt_tokens": 0}
            )

    class RecordingResponses:
        def __init__(self, response):
            self.response = response
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return self.response

    invoker = LiteLLMInvoker.__new__(LiteLLMInvoker)
    fake_response = FakeResponse()
    recorder = RecordingResponses(fake_response)
    invoker.client = SimpleNamespace(responses=recorder)
    invoker.together_client = None
    invoker._api_key = "sk-test"

    prompt = Prompt(
        path="response/search",
        prompt="Summarize with tools.",
        allow_web_search=True,
        computer_use_config={"display_width": 1024, "display_height": 768, "environment": "browser"},
    )

    dialog = SimpleNamespace(
        messages=[
            Message(role=Roles.SYSTEM, content="System", name="system"),
            Message(role=Roles.USER, content="Tell me about markets", name="user"),
        ]
    )

    message = invoker.call(
        dialog,
        prompt,
        model="response-model",
        api_type=APITypes.RESPONSE,
    )

    tools = recorder.last_kwargs["tools"]
    tool_types = {tool["type"] for tool in tools}
    assert "web_search_preview" in tool_types
    assert "computer_use_preview" in tool_types

    comp_tool = next(tool for tool in tools if tool["type"] == "computer_use_preview")
    assert comp_tool["display_width"] == 1024
    assert comp_tool["display_height"] == 768
    assert comp_tool["environment"] == "browser"

    assert message.api_type == APITypes.RESPONSE
    assert message.content == "Final search-backed answer."
