"""Regression tests for native runtime fixes found while rewriting the docs."""

import json
from typing import Any, Dict, List, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

pytest.importorskip("litellm", reason="these tests drive the native runtime")

from lllm import ParseError, Prompt, Tactic, tool  # noqa: E402
from lllm.runtimes.native import Dialog, Role  # noqa: E402
from lllm.runtimes.native.core.dialog import DefaultContextManager  # noqa: E402
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
    agent = Tactic.quick(
        system_prompt="You take notes.",
        model="gpt-4o-mini",
    )
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


def test_default_context_manager_does_not_trim_for_an_unknown_window(caplog):
    manager = DefaultContextManager("openai/fix-native-core-unknown-model")
    dialog = _dialog("Hello.", "What is 2 + 2?")

    with caplog.at_level("WARNING"):
        assert manager.max_tokens is None
        assert manager(dialog) is dialog
    assert "set max_tokens" in caplog.text.lower()


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
