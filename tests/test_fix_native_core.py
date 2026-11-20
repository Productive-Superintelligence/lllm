"""Regression tests for native runtime fixes found while rewriting the docs."""

import base64
import json
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Literal, Optional

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    model_validator,
)

pytest.importorskip("litellm", reason="these tests drive the native runtime")

from lllm import ParseError, Prompt, Tactic, tool  # noqa: E402
from lllm.runtimes.native import (  # noqa: E402
    DefaultContextManager,
    Dialog,
    InvokeCost,
    Modalities,
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
    agent.context_manager = Exact("gpt-4o-mini", max_tokens=40)
    agent.open("notes")
    with ScriptedModel(["Noted."]) as model:
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday"):
            agent.receive(f"Note: on {day} we shipped a small fix.")
            agent.respond()

    dialog = agent.current_dialog
    assert len(dialog.messages) == 9
    # The last call answered the first 8 messages; trimming sent fewer.
    assert len(model.requests[-1].messages) < len(dialog.messages) - 1
    assert dialog.children == []


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
    dialog = _dialog("word " * 10_000, "What now?")  # over any default window

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


def test_registered_models_give_their_window_as_max_input_tokens():
    import litellm

    only_max, with_input = "fixcore/only-max-tokens", "fixcore/with-input"
    for model in (only_max, with_input):
        _forget_window(model)
    try:
        # max_tokens alone may be an output limit, so it is not a window.
        _register(only_max, max_tokens=8192)
        assert DefaultContextManager(only_max).max_tokens is None
        # Unknown at first; found once registered (None is not remembered).
        assert DefaultContextManager(with_input).max_tokens is None
        _register(with_input, max_input_tokens=32_000, max_tokens=4096)
        assert DefaultContextManager(with_input).max_tokens == 32_000
    finally:
        for model in (only_max, with_input):
            litellm.model_cost.pop(model, None)
            _forget_window(model)


def _fake_litellm(monkeypatch, provider: str, info: Dict[str, Any]) -> list:
    """Make LiteLLM know one model of ``provider`` by its unprefixed name."""
    import litellm

    lookups = []

    def get_llm_provider(model, *args, **kwargs):
        return model.split("/", 1)[1], provider, None, None

    def get_model_info(model, custom_llm_provider=None, **kwargs):
        lookups.append((model, custom_llm_provider))
        if model.startswith("huggingface/"):  # the hub URL 404s on the prefix
            raise Exception(f"{model} not found")
        return dict(info, litellm_provider=provider)

    monkeypatch.setattr(litellm, "get_llm_provider", get_llm_provider)
    monkeypatch.setattr(litellm, "get_model_info", get_model_info)
    return lookups


def test_a_hugging_face_window_is_looked_up_without_the_prefix(monkeypatch):
    model = "huggingface/fixcore-org/fixcore-model"
    _forget_window(model)
    # LiteLLM gives the hub's max_position_embeddings as max_tokens.
    lookups = _fake_litellm(monkeypatch, "huggingface", {"max_tokens": 32_768})
    try:
        assert DefaultContextManager(model).max_tokens == 32_768
    finally:
        _forget_window(model)
    assert lookups[0] == ("fixcore-org/fixcore-model", "huggingface")


def test_max_tokens_alone_is_an_output_limit_for_other_providers(monkeypatch):
    model = "gradient_ai/fixcore-model"
    _forget_window(model)
    _fake_litellm(monkeypatch, "gradient_ai", {"max_tokens": 4_096})
    try:
        assert DefaultContextManager(model).max_tokens is None
    finally:
        _forget_window(model)


# -- 3: a message that partly fits keeps the part the marker says ----------


MARKER = "[...earlier content truncated...]\n"
CM_LOGGER = "lllm.runtimes.native.core.dialog"


class Exact(DefaultContextManager):
    """No safety buffer: ``max_tokens`` is exactly the budget."""

    SAFETY_BUFFER = 0


def test_partly_kept_message_keeps_its_end_after_the_marker():
    long = "BEGIN " + " ".join(f"w{i}" for i in range(400)) + " END"
    dialog = _dialog(long, "Short question?")
    trimmed = Exact("gpt-4o-mini", max_tokens=200)(dialog)

    border, last = trimmed.messages[1:]
    assert last.content == "Short question?"
    assert border.content.startswith(MARKER)
    assert border.content.endswith(" END")
    assert "BEGIN" not in border.content
    assert long.endswith(border.content.removeprefix(MARKER))
    assert dialog.messages[1].content == long  # the dialog is never changed


def _photo(size: int) -> str:
    """Base64 data that put_image() takes for a PNG of ``size`` bytes."""
    import random

    data = b"\x89PNG\r\n\x1a\n" + random.Random(size).randbytes(size)
    return base64.b64encode(data).decode()


def _tokens(manager: DefaultContextManager, message: Any) -> int:
    return manager._tokens(message, manager._to_raw(message))


def test_images_count_at_high_detail_not_as_their_data():
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    photo = io.BytesIO()
    Image.new("RGB", (1024, 768), "white").save(photo, format="PNG")
    dialog = _dialog()
    dialog.put_image(_photo(360_000), caption="A photo")  # 480,000 characters
    dialog.put_image(base64.b64encode(photo.getvalue()).decode())
    dialog.put_text("What is in them?")
    manager = DefaultContextManager("gpt-4o-mini")

    assert manager(dialog) is dialog
    for message in dialog.messages[1:3]:  # OpenAI: 765 tokens at 1024x768
        assert 765 <= _tokens(manager, message) < 1_200


def test_an_image_litellm_cannot_size_counts_as_image_tokens():
    from lllm.runtimes.native import Message

    dialog = _dialog("What is in it?")
    dialog.append(
        Message(
            role=Role.USER,
            name="user",
            content="!!not base64!!",
            modality=Modalities.IMAGE,
            metadata={"caption": "A photo"},
        )
    )
    manager = DefaultContextManager("gpt-4o-mini")
    tokens = _tokens(manager, dialog.messages[-1])

    assert manager.IMAGE_TOKENS <= tokens < manager.IMAGE_TOKENS + 50
    assert manager(dialog) is dialog  # counting the dialog does not fail


def test_tool_call_arguments_are_counted():
    dialog = _dialog("Find notes about LLLM.")
    _tool_round(dialog, "Found 3 notes.")
    dialog.messages[2].function_calls[0].arguments = {"query": "lllm " * 400}
    manager = DefaultContextManager("gpt-4o-mini")

    assert _tokens(manager, dialog.messages[2]) > 400  # the content is short


def test_the_newest_message_is_sent_whole_even_over_budget(caplog):
    long = "word " * 500
    dialog = _dialog("An older note.", long)
    with caplog.at_level("WARNING", logger=CM_LOGGER):
        trimmed = Exact("gpt-4o-mini", max_tokens=60)(dialog)
    assert [m.content for m in trimmed.messages] == ["You help.", long]
    assert "newest" in caplog.text

    dialog = _dialog("An older note.")
    dialog.put_image(_photo(1_000), caption="A photo")
    trimmed = Exact("gpt-4o-mini", max_tokens=40)(dialog)
    assert [m.modality for m in trimmed.messages] == [
        Modalities.TEXT,
        Modalities.IMAGE,
    ]
    assert trimmed.messages[1].content == dialog.messages[2].content


def test_a_dropped_image_is_dropped_whole_and_logged(caplog):
    dialog = _dialog()
    dialog.put_image(_photo(1_000), caption="Q3 revenue")
    dialog.put_text("word " * 50)
    # Over 50 tokens are left at the image, enough to cut a text message.
    with caplog.at_level("WARNING", logger=CM_LOGGER):
        trimmed = Exact("gpt-4o-mini", max_tokens=130)(dialog)

    assert [m.modality for m in trimmed.messages] == [Modalities.TEXT] * 2
    assert "dropped 1 image" in caplog.text


def _tool_round(dialog: Dialog, *results: str) -> None:
    """A tool-call reply and one tool result per call, as respond() adds."""
    from lllm.runtimes.native import FunctionCall, Message

    calls = [
        FunctionCall(id=f"call_{i}", name="lookup_order", arguments={"n": i})
        for i in range(len(results))
    ]
    dialog.append(
        Message(
            role=Role.TOOL_CALL,
            name="assistant",
            content="Tool calls:\n\n0. lookup_order",
            function_calls=calls,
        )
    )
    for call, result in zip(calls, results):
        dialog.put_text(
            result,
            name=call.name,
            role=Role.TOOL,
            metadata={"tool_call_id": call.id},
        )


def _roles(dialog: Dialog) -> list:
    return [message.role for message in dialog.messages]


def test_a_tool_call_is_kept_or_dropped_with_its_results():
    dialog = _dialog("Where are orders 1 and 2? I need them soon. " * 10)
    _tool_round(dialog, "Order 1 shipped. " * 20, "Order 2 is packed. " * 20)
    dialog.put_text("And when do they arrive?")
    manager = Exact("gpt-4o-mini", max_tokens=10_000)
    head, _, call, *results, newest = dialog.messages
    tokens = {id(m): manager._count([manager._to_raw(m)]) for m in dialog.messages}

    # Room for the newest message and both results, but not for their call.
    budget = sum(tokens[id(m)] for m in (head, newest, *results))
    budget += tokens[id(call)] // 2
    trimmed = Exact("gpt-4o-mini", max_tokens=budget)(dialog)
    assert _roles(trimmed) == [Role.SYSTEM, Role.USER]

    budget += tokens[id(call)]  # now the whole round fits
    trimmed = Exact("gpt-4o-mini", max_tokens=budget)(dialog)
    assert _roles(trimmed) == [
        Role.SYSTEM,
        Role.TOOL_CALL,
        Role.TOOL,
        Role.TOOL,
        Role.USER,
    ]


def test_a_newest_tool_result_keeps_its_call():
    dialog = _dialog("Where is order 1?")
    _tool_round(dialog, "Order 1 shipped on Monday. " * 30)
    trimmed = Exact("gpt-4o-mini", max_tokens=60)(dialog)

    assert _roles(trimmed) == [Role.SYSTEM, Role.TOOL_CALL, Role.TOOL]


def _trims(manager: DefaultContextManager, words: int) -> bool:
    dialog = _dialog("word " * words, "What now?")  # about words + 20 tokens
    return manager(dialog) is not dialog


def test_the_default_buffer_is_5000_tokens_or_half_a_smaller_budget():
    def default(max_tokens: int) -> DefaultContextManager:
        return DefaultContextManager("gpt-4o-mini", max_tokens=max_tokens)

    # 16,000 left after 5,000; an uncapped half or quarter would leave less.
    assert not _trims(default(21_000), 15_880)
    # 4,000 left after half; a quarter would leave 6,000.
    assert _trims(default(8_000), 4_990)
    assert not _trims(default(8_000), 3_900)
    # 800 left after half: small budgets still work.
    assert not _trims(default(1_600), 700)


def test_a_safety_buffer_that_is_set_is_used_as_is():
    class Roomy(DefaultContextManager):
        SAFETY_BUFFER = 6_000  # more than half of 8,000

    # 2,000 left; a value capped at half the budget would leave 4,000.
    assert _trims(Roomy("gpt-4o-mini", max_tokens=8_000), 2_990)

    roomy = DefaultContextManager("gpt-4o-mini", max_tokens=8_000)
    roomy.SAFETY_BUFFER = 6_000
    assert _trims(roomy, 2_990)

    # On the class itself, the one-line way to set it for config-built agents.
    default = DefaultContextManager.SAFETY_BUFFER
    try:
        DefaultContextManager.SAFETY_BUFFER = 6_000
        assert _trims(DefaultContextManager("gpt-4o-mini", max_tokens=8_000), 2_990)
    finally:
        DefaultContextManager.SAFETY_BUFFER = default
    assert DefaultContextManager.SAFETY_BUFFER == 5_000  # as in 0.4.0


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


GLOBAL = {"model_name": "gpt-4o-mini", "max_llm_recall": 2}


@pytest.mark.parametrize("by_alias", [False, True], ids=["dump", "serialize_by_alias"])
def test_a_config_with_an_alias_generator_is_read_by_field_name(by_alias):
    from pydantic.alias_generators import to_camel

    camel = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, serialize_by_alias=by_alias
    )

    class CamelEntry(BaseModel):
        model_config = camel
        name: str
        system_prompt: str

    class CamelConfig(BaseModel):
        model_config = camel
        global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
        agent_configs: List[CamelEntry]

    config = CamelConfig.model_validate(
        {
            "global": GLOBAL,
            "agentConfigs": [{"name": "echo", "systemPrompt": "You echo."}],
        }
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall, echo.system_prompt.prompt) == (
        "gpt-4o-mini",
        2,
        "You echo.",
    )


@pytest.mark.parametrize(
    ("alias", "key"),
    [
        ({"validation_alias": "global"}, "global"),
        ({"validation_alias": AliasChoices("defaults", "global")}, "global"),
        ({"validation_alias": AliasPath("global")}, "global"),
        ({"validation_alias": AliasChoices("defaults", AliasPath("global"))}, "global"),
        ({"serialization_alias": "global"}, "global_"),
    ],
    ids=["validation_alias", "alias_choices", "alias_path", "path_choice", "serialization"],
)
def test_the_field_aliased_global_is_the_global_section(alias, key):
    class Config(BaseModel):
        global_: Dict[str, Any] = Field(default_factory=dict, **alias)
        agent_configs: List[AgentEntry]

    config = Config.model_validate(
        {key: GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)


class TrimSettings(BaseModel):
    type: Optional[str] = "default"
    max_tokens: Optional[int] = None


class TeamEntry(BaseModel):
    name: str
    system_prompt: str
    model_name: Optional[str] = None
    context_manager: Optional[TrimSettings] = None


class TeamConfig(BaseModel):
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[TeamEntry]
    max_workers: Optional[int] = None


class Team(Tactic):
    name = "fix_team"
    agent_group = ["writer", "editor", "archivist"]


def test_unset_none_fields_fall_back_to_global_and_set_ones_apply():
    config = TeamConfig.model_validate(
        {
            "global": {
                "model_name": "gpt-4o-mini",
                "context_manager": {"max_tokens": 64_000},
            },
            "agent_configs": [
                {
                    "name": "writer",
                    "system_prompt": "You write.",
                    "context_manager": {"type": "default"},  # max_tokens unset
                },
                {
                    "name": "editor",
                    "system_prompt": "You edit.",
                    "model_name": "gpt-4o",
                    "context_manager": {"max_tokens": 9_000},
                },
                {
                    "name": "archivist",
                    "system_prompt": "You keep everything.",
                    "context_manager": {"type": None},  # set: no trimming
                },
            ],
        }
    )
    team = Team(config)
    writer, editor, archivist = (
        team.agents[name] for name in ("writer", "editor", "archivist")
    )

    assert (writer.model, writer.context_manager.max_tokens) == (
        "gpt-4o-mini",
        64_000,
    )
    assert (editor.model, editor.context_manager.max_tokens) == ("gpt-4o", 9_000)
    assert (archivist.model, archivist.context_manager) == ("gpt-4o-mini", None)
    assert team._max_workers == 4  # unset, so the default


class TeamEntries(RootModel[List[TeamEntry]]):
    pass


def test_agent_entries_in_a_root_model_fall_back_to_global():
    class Config(BaseModel):
        global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
        agent_configs: TeamEntries

    config = Config.model_validate(
        {"global": GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    echo = Echo(config).agents["echo"]
    assert (echo.model, echo.max_llm_recall) == ("gpt-4o-mini", 2)


class ByName(BaseModel):
    tactic_type: ClassVar[str] = "fix_echo"
    global_: Dict[str, Any] = Field(default_factory=dict, alias="global")
    agent_configs: List[AgentEntry]

    @property
    def max_workers(self) -> int:
        return 2


def test_class_vars_and_properties_of_a_config_are_read():
    from lllm.runtimes.native import Runtime
    from lllm.runtimes.native.core.tactic_registry import (
        build_tactic,
        register_tactic_class,
    )

    runtime = Runtime()
    register_tactic_class(Echo, runtime=runtime)
    config = ByName.model_validate(
        {"global": GLOBAL, "agent_configs": [{"name": "echo", "system_prompt": "Hi."}]}
    )
    tactic = build_tactic(config, runtime=runtime)  # by its tactic_type

    assert isinstance(tactic, Echo)
    assert tactic._max_workers == 2


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
    # Invalid JSON: a ParseError with the decoder's message, as before.
    with pytest.raises(ParseError) as info:
        TRIAGE.parse("billing")
    assert str(info.value) == "Expecting value: line 1 column 1 (char 0)"
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


def test_format_with_a_json_schema_needs_a_json_object():
    ask = Prompt(path="fix/schema", prompt="Answer.", format={"type": "object"})

    assert ask.parse('{"answer": 42}') == {"answer": 42}
    with pytest.raises(ParseError) as info:
        ask.parse("[42]")
    assert str(info.value) == "The reply must be a JSON object"  # no "."


# -- 6: ``parsed`` holds exactly what ``format=`` or the parser produced ----


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: Decimal
    note: str = Field(default="", exclude=True)
    paid_by: str = Field(alias="paidBy")
    reference: str = Field(validation_alias=AliasChoices("reference", "ref"))

    @computed_field
    @property
    def label(self) -> str:
        return f"{self.vendor} {self.amount}"


def test_parsed_is_the_reply_json_and_validates_back_into_the_model():
    ask = Prompt(path="fix/invoice", prompt="Extract: {text}", format=Invoice)
    reply = (
        '{"vendor": "ACME", "amount": 1250.10, "note": "rush",'
        ' "paidBy": "card", "ref": "INV-7"}'
    )
    parsed = ask.parse(reply)

    assert parsed == json.loads(reply)  # a number stays a number
    invoice = Invoice.model_validate(parsed)  # no computed "label" in it
    assert (invoice.amount, invoice.note, invoice.label) == (
        Decimal("1250.1"),
        "rush",
        "ACME 1250.1",
    )


def _parse(model: Any, reply: str) -> Dict[str, Any]:
    return Prompt(path="fix/format", prompt="Answer.", format=model).parse(reply)


def test_parsed_is_the_whole_reply_object_without_defaults():
    class Plain(BaseModel):
        team: str
        urgent: bool = False
        is_global: bool = Field(default=False, alias="global")

    reply = '{"team": "bugs", "global": true, "confidence": 0.9}'
    parsed = _parse(Plain, reply)

    assert parsed == json.loads(reply)  # unfiltered; no defaults added
    assert Plain.model_validate(parsed) == Plain(team="bugs", **{"global": True})


class Envelope(BaseModel):
    team: str

    @model_validator(mode="before")
    @classmethod
    def unwrap(cls, data: Any) -> Any:
        return data.get("data", data) if isinstance(data, dict) else data


class Renamed(BaseModel):
    team: str

    @model_validator(mode="wrap")
    @classmethod
    def rename(cls, data: Any, handler: Any) -> Any:
        if isinstance(data, dict) and "group" in data:
            data = {"team": data["group"]}
        return handler(data)


class Located(BaseModel):
    name: str = Field(validation_alias=AliasPath("data", "name"))


@pytest.mark.parametrize(
    ("model", "reply"),
    [
        (RootModel[Dict[str, int]], '{"alice": 3, "bob": 5}'),
        (Envelope, '{"data": {"team": "bugs"}}'),
        (Renamed, '{"group": "bugs"}'),
        (Located, '{"data": {"name": "Ada"}}'),
    ],
    ids=["root_model", "before_validator", "wrap_validator", "alias_path"],
)
def test_parsed_validates_back_however_the_model_reads_it(model, reply):
    parsed = _parse(model, reply)

    assert parsed == json.loads(reply)
    assert model.model_validate(parsed) == model.model_validate_json(reply)


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


WRITER = {"model_name": "gpt-4o-mini", "system_prompt": "You write."}


@pytest.mark.parametrize("value", [[], "", 0, False, ["style"]])
def test_extra_settings_must_be_a_mapping(value):
    from lllm.runtimes.native import AgentSpec

    with pytest.raises(TypeError, match="'extra_settings' must be a mapping"):
        AgentSpec.from_config("writer", {**WRITER, "extra_settings": value})


def test_extra_settings_take_any_mapping_and_values_deepcopy_cannot_copy():
    import threading
    from types import MappingProxyType

    from lllm.runtimes.native import AgentSpec, Runtime

    lock = threading.Lock()
    settings = MappingProxyType({"lock": lock, "style": "terse"})
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    first, second = (spec.build(Runtime(), object()) for _ in range(2))

    assert first.extra_settings == {"lock": lock, "style": "terse"}
    assert first.extra_settings["lock"] is lock  # shared, not copied
    assert first.extra_settings is not second.extra_settings
    unset = AgentSpec.from_config("writer", {**WRITER, "extra_settings": None})
    assert unset.extra_settings == {}


class NoCopies:
    """A live object, such as a client, that must never be deep-copied."""

    def __deepcopy__(self, memo):
        raise AssertionError("deep-copied a live object")


def test_extra_settings_copy_plain_data_and_share_other_objects():
    from lllm.runtimes.native import AgentSpec, Runtime

    client = NoCopies()
    settings = {"client": client, "tags": ["a"], "limits": {"words": [100]}}
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": settings})
    first, second = (spec.build(Runtime(), object()) for _ in range(2))

    first.extra_settings["tags"].append("b")
    first.extra_settings["limits"]["words"].append(200)
    settings["tags"].append("c")  # the config given, after parsing
    for owner in (second, spec):
        assert getattr(owner, "extra_settings") == {
            "client": client,
            "tags": ["a"],
            "limits": {"words": [100]},
        }
    assert first.extra_settings["client"] is client


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_an_sdk_client_in_extra_settings_is_shared_cleanly():
    import gc

    openai = pytest.importorskip("openai")
    from lllm.runtimes.native import AgentSpec, Runtime

    client = openai.OpenAI(api_key="sk-test", base_url="http://localhost:9")
    spec = AgentSpec.from_config("writer", {**WRITER, "extra_settings": {"client": client}})
    agents = [spec.build(Runtime(), object()) for _ in range(2)]
    gc.collect()  # a half-copied client would complain here

    assert all(agent.extra_settings["client"] is client for agent in agents)


# -- a failed agent call keeps its record ------------------------------------


RATE = Prompt(path="fix/rate-reply", prompt="Rate 1-5: {review}", format=Rating)
NOT_JSON = ["Four stars.", "Four stars, really."]


def _replies(call: Any) -> list:
    return [t.message.content for ts in call.invoke_traces.values() for t in ts]


def test_a_failed_respond_carries_its_session():
    from lllm.runtimes.native import AgentException

    agent = Tactic.quick(
        system_prompt="You rate reviews.", model="gpt-4o-mini", max_exception_retry=1
    )
    agent.open("rate")
    agent.receive_prompt(RATE, {"review": "Fast shipping, crushed box."})
    with ScriptedModel(NOT_JSON), pytest.raises(AgentException) as info:
        agent.respond()

    call = info.value.agent_session
    assert (call.state, call.exception_retries_count) == ("failure", 1)
    assert _replies(call) == NOT_JSON


class Rate(Tactic):
    name = "fix_rate"
    agent_group = ["rater"]

    def call(self, review: str) -> dict:
        rater = self.agents["rater"]
        rater.open("rate")
        rater.receive_prompt(RATE, {"review": review})
        return rater.respond().parsed


def test_a_tactic_session_records_the_agent_call_that_failed():
    from lllm.runtimes.native import AgentException

    sessions = []

    class Keep:
        def on_tactic_failure(self, *, session, **_):
            sessions.append(session)

    rate = Rate(
        {
            "global": {"model_name": "gpt-4o-mini", "max_exception_retry": 1},
            "agent_configs": [{"name": "rater", "system_prompt": "You rate."}],
        },
        observer=Keep(),
    )
    with ScriptedModel(NOT_JSON), pytest.raises(AgentException):
        rate("Fast shipping, crushed box.")

    (session,) = sessions
    (call,) = session.agent_sessions["rater"]
    assert (call.state, _replies(call)) == ("failure", NOT_JSON)


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
        model_args={"seed": 7, "temperature": 0.9},
    )

    assert agent.name == "assistant"
    assert agent.max_llm_recall == 2
    assert agent.max_exception_retry == 1
    assert agent.max_interrupt_steps == 2
    assert agent.max_rate_limit_retries == 0
    assert agent.extra_settings == {"team": "docs"}
    # A keyword wins over the same key in model_args=.
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
