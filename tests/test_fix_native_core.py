"""Regression tests for native runtime fixes found while rewriting the docs."""

import base64
import json
from typing import Any, Dict, List, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

pytest.importorskip("litellm", reason="these tests drive the native runtime")

from lllm import ParseError, Prompt, Tactic, tool  # noqa: E402
from lllm.runtimes.native import (  # noqa: E402
    DefaultContextManager,
    Dialog,
    InvokeCost,
    Role,
)
from lllm.testing import ScriptedModel, ToolCall  # noqa: E402


# -- 1: the agent loop's working copies stay out of the dialog tree --------


class Rating(BaseModel):
    stars: int


@tool(description="Look up a word")
def define(word: str) -> str:
    return f"{word}: damaged by pressure"


def test_respond_keeps_working_copies_out_of_the_dialog_tree():
    rate = Prompt(
        path="fix/rate",
        prompt="Rate 1-5: {review}",
        format=Rating,
        function_list=[define],
    )
    agent = Tactic.quick(system_prompt="You rate reviews.", model="gpt-4o-mini")
    agent.open("rate")
    agent.receive_prompt(rate, {"review": "Crushed box."})
    replies = [ToolCall("define", {"word": "crushed"}), "Four.", '{"stars": 4}']
    with ScriptedModel(replies):
        session = agent.respond(return_session=True)

    assert session.exception_retries_count == 1
    assert len(session.tool_calls) == 1
    dialog = agent.current_dialog
    # system, prompt, tool call, tool result, accepted reply; no repairs
    assert [m.role for m in dialog.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.TOOL_CALL,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert dialog.children == []
    assert dialog.tree_node.children_ids == []
    assert dialog.to_dict()["children"] == []
    assert dialog.tree_overview().count("\n") == 0

    agent.fork("rate", "retry")
    assert dialog.children == [agent.dialogs["retry"]]
    assert dialog.tree_node.children_ids == [agent.dialogs["retry"].dialog_id]


def test_context_manager_copies_stay_out_of_the_dialog_tree():
    agent = Tactic.quick(system_prompt="You take notes.", model="gpt-4o-mini")
    agent.context_manager = DefaultContextManager("gpt-4o-mini", max_tokens=5_030)
    agent.open("notes")
    with ScriptedModel(["Noted."]) as model:
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            agent.receive(f"Note: on {day} we shipped a small fix.")
            agent.respond()

    assert len(model.requests[-1].messages) < 2 * 4 + 1  # it did trim
    assert agent.current_dialog.children == []
    assert len(agent.current_dialog.messages) == 9


def test_saved_dialogs_grow_linearly_with_turns():
    def saved_size(turns: int) -> int:
        agent = Tactic.quick(system_prompt="You echo.", model="gpt-4o-mini")
        agent.open("echo")
        with ScriptedModel(["ok"]):
            for turn in range(turns):
                agent.receive(f"turn {turn}")
                agent.respond()
        return len(json.dumps(agent.current_dialog.to_dict()))

    assert saved_size(40) < 2.2 * saved_size(20)


def test_working_copy_names_its_dialog_as_parent():
    dialog = Dialog(owner="agent")
    dialog.put_text("You help.", name="system", role=Role.SYSTEM)
    copy = dialog._copy()

    assert copy.parent is dialog
    assert copy.tree_node.parent_id == dialog.dialog_id
    assert copy.depth == 1
    assert [m.content for m in copy.messages] == ["You help."]
    assert dialog.children == []


# -- 2: the context manager's budget is the context window -----------------


def _dialog(*texts: str) -> Dialog:
    dialog = Dialog(owner="agent")
    dialog.put_text("You help.", name="system", role=Role.SYSTEM)
    for text in texts:
        dialog.put_text(text)
    return dialog


def test_default_context_manager_budget_is_the_context_window():
    assert DefaultContextManager("gpt-4o-mini").max_tokens == 128_000
    assert DefaultContextManager("gpt-4o-mini", max_tokens=9_000).max_tokens == 9_000

    # About 15,000 tokens: more than gpt-4o-mini's output limit (16,384)
    # leaves after the safety buffer, far less than its 128,000 window.
    dialog = _dialog("word " * 15_000, "What now?")
    assert DefaultContextManager("gpt-4o-mini")(dialog) is dialog


def _forget_window(model: str) -> None:
    from lllm.runtimes.native.core import dialog as dialog_module

    dialog_module._CONTEXT_WINDOWS.pop(model, None)
    dialog_module._UNKNOWN_WINDOWS.discard(model)


def test_an_unknown_window_is_not_trimmed_and_warned_about_once(caplog):
    model = "openai/fix-native-core-unknown-model"
    _forget_window(model)
    manager = DefaultContextManager(model)
    dialog = _dialog("Hello.", "What is 2 + 2?")

    with caplog.at_level("WARNING", logger="lllm.runtimes.native.core.dialog"):
        assert manager.max_tokens is None
        assert manager(dialog) is dialog
        assert DefaultContextManager(model).max_tokens is None
    warnings = [r for r in caplog.records if "context window" in r.getMessage()]
    assert len(warnings) == 1
    assert "set max_tokens" in warnings[0].getMessage().lower()


def _register(model: str, **limits: int) -> None:
    import litellm

    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                **limits,
            }
        }
    )


def test_max_tokens_is_the_window_when_no_output_limit_is_given():
    import litellm

    only_max, with_output = "fixcore/only-max-tokens", "fixcore/with-output"
    for model in (only_max, with_output):
        _forget_window(model)
    try:
        # Unknown at first; found once registered (None is not remembered).
        assert DefaultContextManager(only_max).max_tokens is None
        _register(only_max, max_tokens=8192)
        assert DefaultContextManager(only_max).max_tokens == 8192
        # With an output limit beside it, max_tokens is that output limit.
        _register(with_output, max_tokens=4096, max_output_tokens=4096)
        assert DefaultContextManager(with_output).max_tokens is None
    finally:
        for model in (only_max, with_output):
            litellm.model_cost.pop(model, None)
            _forget_window(model)


# -- 3: a message that partly fits keeps the part the marker says ----------


MARKER = "[...earlier content truncated...]\n"


def test_partly_kept_message_keeps_its_end_after_the_marker():
    long = "BEGIN " + " ".join(f"w{i}" for i in range(400)) + " END"
    dialog = _dialog(long, "Short question?")
    trimmed = DefaultContextManager("gpt-4o-mini", max_tokens=5_200)(dialog)

    border, last = trimmed.messages[1:]
    assert last.content == "Short question?"
    assert border.content.startswith(MARKER)
    assert border.content.endswith(" END")
    assert "BEGIN" not in border.content
    assert long.endswith(border.content.removeprefix(MARKER))
    assert dialog.messages[1].content == long  # the dialog is never changed


def test_images_are_dropped_not_cut():
    import base64

    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8
    dialog = _dialog()
    dialog.put_image(base64.b64encode(png).decode(), caption="A chart")
    dialog.put_text("What does it show?")
    trimmed = DefaultContextManager("gpt-4o-mini", max_tokens=5_200)(dialog)

    assert [m.content for m in trimmed.messages] == [
        "You help.",
        "What does it show?",
    ]


# -- 4: a Pydantic config keeps its aliased ``global`` section ---------------


class AgentEntry(BaseModel):
    name: str
    system_prompt: str


class EchoConfig(BaseModel):
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[AgentEntry]


class Echo(Tactic):
    name = "fix_echo"
    agent_group = ["echo"]

    def call(self, text: str) -> str:
        echo = self.agents["echo"]
        echo.open("echo")
        echo.receive(text)
        return echo.respond().content


def test_pydantic_config_keeps_its_global_section():
    config = EchoConfig.model_validate(
        {
            "global": {"model_name": "gpt-4o-mini", "max_llm_recall": 2},
            "agent_configs": [{"name": "echo", "system_prompt": "You echo."}],
        }
    )
    tactic = Echo(config)

    echo = tactic.agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)
    with ScriptedModel(["hello"]):
        assert tactic("hello") == "hello"


# -- 5: ``format=`` validates the reply against the model --------------------


class Triage(BaseModel):
    team: Literal["billing", "bugs", "accounts"]
    priority: int


TRIAGE = Prompt(path="fix/triage", prompt="Route: {ticket}", format=Triage)
NOT_A_TEAM = "team: Input should be 'billing', 'bugs' or 'accounts'"


def test_format_repairs_replies_whose_fields_do_not_fit():
    agent = Tactic.quick(system_prompt="You route tickets.", model="gpt-4o-mini")
    agent.open("triage")
    agent.receive_prompt(TRIAGE, {"ticket": "Wrong company name on my invoice."})
    replies = ['{"team": "invoicing", "priority": 3}', '{"team": "billing", "priority": 3}']
    with ScriptedModel(replies) as model:
        session = agent.respond(return_session=True)

    assert session.exception_retries_count == 1
    (error,) = session.invoke_traces[0][0].execution_errors
    assert isinstance(error, ParseError)
    assert str(error) == f"The reply does not match Triage: {NOT_A_TEAM}"
    assert model.requests[1].text == (
        f"Error: The reply does not match Triage: {NOT_A_TEAM}. Please fix."
    )
    assert session.delivery.parsed == {"team": "billing", "priority": 3}


def test_format_lists_every_problem_with_its_location():
    with pytest.raises(ParseError) as info:
        TRIAGE.parse('{"team": "sales"}')
    assert str(info.value) == (
        f"The reply does not match Triage: {NOT_A_TEAM}; priority: Field required"
    )

    with pytest.raises(ParseError, match=r"Triage: Input should be an object$"):
        TRIAGE.parse("[1, 2]")
    with pytest.raises(json.JSONDecodeError):  # invalid JSON, as before
        TRIAGE.parse("billing")


def test_format_with_a_json_schema_needs_a_json_object():
    ask = Prompt(path="fix/schema", prompt="Answer.", format={"type": "object"})

    assert ask.parse('{"answer": 42}') == {"answer": 42}
    with pytest.raises(ParseError, match="JSON object"):
        ask.parse("[42]")


# -- 6: ``parsed`` holds exactly what ``format=`` or the parser produced ----


class Ticket(BaseModel):
    team: str
    urgent: bool = False
    is_global: bool = Field(default=False, alias="global")


def test_format_results_hold_exactly_the_model_fields():
    ask = Prompt(path="fix/ticket", prompt="Route: {ticket}", format=Ticket)
    parsed = ask.parse('{"team": "bugs", "global": true, "confidence": 0.9}')

    assert parsed == {"team": "bugs", "urgent": False, "global": True}
    assert Ticket.model_validate(parsed).is_global is True


class StrictTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: str
    priority: int


def as_strict_triage(text: str, **kwargs) -> dict:
    return StrictTriage.model_validate_json(text).model_dump()


class StrictRoute(Tactic):
    name = "fix_strict_route"
    agent_group = ["router"]
    output_model = StrictTriage

    def call(self, ticket: str) -> StrictTriage:
        router = self.agents["router"]
        router.open("route")
        router.receive_prompt(self.prompt, {"ticket": ticket})
        return router.respond().parsed


@pytest.mark.parametrize(
    "contract",
    [{"format": StrictTriage}, {"parser": as_strict_triage}],
    ids=["format", "parser"],
)
def test_parsed_validates_into_a_model_that_forbids_extra_fields(contract):
    route = StrictRoute(
        {
            "global": {"model_name": "gpt-4o-mini"},
            "agent_configs": [{"name": "router", "system_prompt": "You route."}],
        }
    )
    route.prompt = Prompt(path="fix/strict", prompt="Route: {ticket}", **contract)
    with ScriptedModel(['{"team": "billing", "priority": 2}']):
        result = route("I was charged twice.")

    assert result == StrictTriage(team="billing", priority=2)


def test_parser_results_are_used_as_returned():
    def percent(text: str, **kwargs) -> float:
        return float(text.rstrip("%"))

    def fields(text: str, **kwargs) -> dict:
        return {"label": text.strip()}

    assert Prompt(path="fix/percent", prompt="?", parser=percent).parse("37%") == {
        "value": 37.0
    }
    assert Prompt(path="fix/fields", prompt="?", parser=fields).parse(" bug ") == {
        "label": "bug"
    }
    assert Prompt(path="fix/plain", prompt="?").parse("hello") == {}


# -- 7: images keep their media type; overview() shows a placeholder --------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


PNG = b"\x89PNG\r\n\x1a\n" + bytes(2040)
IMAGES = {
    "image/png": PNG,
    "image/jpeg": b"\xff\xd8\xff\xe0" + bytes(60),
    "image/webp": b"RIFF\x24\x00\x00\x00WEBPVP8 " + bytes(60),
    "image/gif": b"GIF89a" + bytes(60),
}


@pytest.mark.parametrize("media_type", sorted(IMAGES))
def test_images_are_sent_with_their_media_type(media_type):
    from lllm.runtimes.native.invokers.litellm import LiteLLMInvoker

    dialog = Dialog(owner="agent")
    dialog.put_image(_b64(IMAGES[media_type]), caption="A picture")
    (message,) = LiteLLMInvoker({})._convert_dialog(dialog)

    caption, image = message["content"]
    assert caption == {"type": "text", "text": "A picture"}
    url = image["image_url"]["url"]
    assert url == f"data:{media_type};base64,{_b64(IMAGES[media_type])}"


def test_put_image_rejects_riff_data_that_is_not_webp():
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(60)
    with pytest.raises(ValueError, match="PNG/JPEG/WebP/GIF"):
        Dialog(owner="agent").put_image(_b64(wav))


def test_overview_shows_an_image_as_type_size_and_caption():
    dialog = _dialog()
    dialog.put_image(_b64(PNG), caption="Revenue by quarter")
    dialog.put_image(_b64(IMAGES["image/gif"]))
    dialog.put_text("What is the trend?")

    assert dialog.overview().split("\n\n") == [
        "[0. system (developer)]: You help.",
        "[1. user (user)]: <image/png, 2.0 KB> Revenue by quarter",
        "[2. user (user)]: <image/gif, 66 B>",
        "[3. user (user)]: What is the trend?",
    ]


# -- 8: ``extra_settings`` from the config reach the agent -------------------


class Styled(Tactic):
    name = "fix_styled"
    agent_group = ["writer"]

    def call(self, key: str) -> Any:
        settings = self.agents["writer"].extra_settings
        settings["calls"] = settings.get("calls", 0) + 1  # per call
        return settings[key]


def test_extra_settings_reach_the_agent():
    styled = Styled(
        {
            "global": {
                "model_name": "gpt-4o-mini",
                "extra_settings": {"style": "terse", "audience": "developers"},
            },
            "agent_configs": [
                {
                    "name": "writer",
                    "system_prompt": "You write.",
                    "extra_settings": {"style": "formal"},
                }
            ],
        }
    )

    assert styled.agents["writer"].extra_settings == {
        "style": "formal",
        "audience": "developers",
    }
    assert styled("style") == "formal"
    assert styled("calls") == 1  # every call's agent gets its own copy
    assert styled("calls") == 1
    assert Tactic.quick(model="gpt-4o-mini").extra_settings == {}


# -- 9: context managers and stream handlers are exported -------------------


def test_extension_base_classes_are_exported_from_the_native_runtime():
    import lllm.runtimes.native as native
    from lllm.runtimes.native.core import dialog
    from lllm.runtimes.native.invokers import base

    exported = {
        "BaseInvoker": base.BaseInvoker,
        "BaseStreamHandler": base.BaseStreamHandler,
        "ContextManager": dialog.ContextManager,
        "DefaultContextManager": dialog.DefaultContextManager,
    }
    for name, value in exported.items():
        assert getattr(native, name) is value
        assert name in native.__all__


# -- 10: str(InvokeCost) shows rates per million tokens ----------------------


@pytest.mark.parametrize(
    ("input_rate", "output_rate", "shown"),
    [
        (0.15e-6, 0.6e-6, "In: $0.15/1M tok, Out: $0.60/1M tok"),  # gpt-4o-mini
        (0.0375e-6, 0.15e-6, "In: $0.0375/1M tok, Out: $0.15/1M tok"),
        (15e-6, 75e-6, "In: $15.00/1M tok, Out: $75.00/1M tok"),
    ],
)
def test_invoke_cost_shows_rates_per_million_tokens(input_rate, output_rate, shown):
    cost = InvokeCost(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
    )
    assert f" | Rates: [{shown}]\n" in str(cost)


def test_reply_cost_shows_the_model_rates():
    with ScriptedModel(["Hi."]):
        reply = Tactic.quick("Hello", model="gpt-4o-mini")
    assert "Rates: [In: $0.15/1M tok, Out: $0.60/1M tok]" in str(reply.cost)


# -- 11: Tactic.quick() takes agent settings as agent settings --------------


def test_quick_sets_agent_fields_and_sends_the_rest_as_model_args():
    agent = Tactic.quick(
        system_prompt="You answer briefly.",
        model="gpt-4o-mini",
        max_llm_recall=2,
        max_exception_retry=1,
        max_interrupt_steps=2,
        max_rate_limit_retries=0,
        extra_settings={"team": "docs"},
        temperature=0.1,
        model_args={"seed": 7},
    )

    assert agent.name == "assistant"
    assert agent.max_llm_recall == 2
    assert agent.max_exception_retry == 1
    assert agent.max_interrupt_steps == 2
    assert agent.max_rate_limit_retries == 0
    assert agent.extra_settings == {"team": "docs"}
    assert agent.model_args == {"seed": 7, "temperature": 0.1}

    agent.open("chat")
    agent.receive("Hi.")
    with ScriptedModel(["Hello."]):
        session = agent.respond(return_session=True)
    assert session.max_llm_recall == 2
    assert session.invoke_traces[0][0].model_args == {
        "seed": 7,
        "temperature": 0.1,
        "drop_params": True,
        "stream": False,
    }


def test_quick_takes_a_stream_handler_and_a_name():
    from lllm.runtimes.native import BaseStreamHandler

    class Collect(BaseStreamHandler):
        def __init__(self):
            self.pieces = []

        def handle_chunk(self, chunk_content, chunk_response):
            self.pieces.append(chunk_content)

    collect = Collect()
    with ScriptedModel(["Hello there."]):
        reply = Tactic.quick(
            "Hi", model="gpt-4o-mini", stream_handler=collect, name="greeter"
        )

    assert reply.name == "greeter"
    assert "".join(collect.pieces) == reply.content == "Hello there."


# -- 12: Responses API usage counts prompt and completion tokens ------------


def test_responses_api_usage_counts_prompt_and_completion_tokens(monkeypatch):
    from types import SimpleNamespace

    from litellm.types.llms.openai import ResponseAPIUsage

    from lllm.runtimes.native import APITypes
    from lllm.runtimes.native.invokers import litellm as invoker_module

    usage = ResponseAPIUsage(
        input_tokens=1200,
        output_tokens=300,
        total_tokens=1500,
        input_tokens_details={"cached_tokens": 1000},
        output_tokens_details={"reasoning_tokens": 200},
    )
    response = SimpleNamespace(
        output=[],
        output_text="Paris.",
        usage=usage,
        reasoning=None,
        _hidden_params={"response_cost": 0.00036},
    )
    monkeypatch.setattr(invoker_module, "responses_api", lambda **kwargs: response)

    agent = Tactic.quick(
        system_prompt="Be brief.", model="gpt-4o-mini", api_type=APITypes.RESPONSE
    )
    agent.open("q")
    agent.receive("What is the capital of France?")
    cost = agent.respond().cost

    assert (cost.prompt_tokens, cost.completion_tokens, cost.total_tokens) == (
        1200,
        300,
        1500,
    )
    assert (cost.cached_prompt_tokens, cost.reasoning_tokens) == (1000, 200)
    assert cost.prompt_cost == pytest.approx(1200 * 0.15e-6)
    assert cost.completion_cost == pytest.approx(300 * 0.6e-6)
    assert cost.cost == pytest.approx(0.00036)
