"""Regression tests for native prompt fixes (review findings 5, 8, 12, 17, 32)."""

import json
import sys
import textwrap
from collections import Counter
from typing import List, Optional, Union

import pytest
from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from lllm import Tactic
from lllm.runtimes.native import (
    Dialog,
    FunctionCall,
    Message,
    Prompt,
    Role,
    Runtime,
    tactic_as_function,
    tool,
)
from lllm.runtimes.native.core.config import load_package
from lllm.runtimes.native.core.const import InvokeResult
from lllm.runtimes.native.core.native_config import AgentSpec


def write(root, files):
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


# -- 5: recursive models in tool schemas ------------------------------------


class Node(BaseModel):
    value: int
    children: List["Node"] = []


class Bin(BaseModel):
    value: int
    left: Optional["Bin"] = None
    right: Optional["Bin"] = None


class Ping(BaseModel):
    pong: Optional["Pong"] = None


class Pong(BaseModel):
    ping: Optional[Ping] = None


Ping.model_rebuild()


class Tree(BaseModel):
    value: int
    children: List["Tree"]


Nested = TypeAliasType("Nested", Union[List["Nested"], int])


class Num(BaseModel):
    value: float


class Var(BaseModel):
    name: str


class BinOp(BaseModel):
    op: str
    left: "Expr"
    right: "Expr"


class Neg(BaseModel):
    operand: "Expr"


class Call(BaseModel):
    func: str
    args: List["Expr"]


class IfExpr(BaseModel):
    cond: "Expr"
    then: "Expr"
    otherwise: "Expr"


class Let(BaseModel):
    name: str
    value: "Expr"
    body: "Expr"


Expr = Union[Num, Var, BinOp, Neg, Call, IfExpr, Let]
EXPR_NODES = ["BinOp", "Neg", "Call", "IfExpr", "Let"]
for _node in (BinOp, Neg, Call, IfExpr, Let):
    _node.model_rebuild()


def assert_no_refs(properties):
    text = json.dumps(properties)
    assert '"$ref"' not in text and '"$defs"' not in text


def spelled_out(schema):
    """Titles of the models written out in full in *schema*, with repeats."""
    if isinstance(schema, list):
        return [title for item in schema for title in spelled_out(item)]
    if not isinstance(schema, dict):
        return []
    own = [schema["title"]] if "properties" in schema and "title" in schema else []
    return own + [title for value in schema.values() for title in spelled_out(value)]


def test_recursive_models_are_expanded_once_without_dangling_refs():
    @tool(description="Walk a tree")
    def walk(tree: Node) -> str:
        return "ok"

    tree = walk.properties["tree"]

    assert_no_refs(walk.properties)
    assert tree["properties"]["value"]["type"] == "integer"
    assert tree["properties"]["children"]["items"] == {"type": "object", "title": "Node"}


def test_branching_recursive_models_build_small_schemas():
    @tool(description="Walk a binary tree")
    def walk_bin(tree: Bin) -> str:
        return "ok"

    # The depth-capped expansion produced a 270 MB schema here.
    assert len(json.dumps(walk_bin.properties)) < 1000
    assert_no_refs(walk_bin.properties)
    assert walk_bin.properties["tree"]["properties"]["left"]["anyOf"] == [
        {"type": "object", "title": "Bin"},
        {"type": "null"},
    ]


def test_mutually_recursive_models_and_aliases_are_cut_on_the_repeat():
    @tool(description="Volley")
    def volley(ping: Ping, nested: Nested) -> str:
        return "ok"

    pong = volley.properties["ping"]["properties"]["pong"]["anyOf"][0]

    assert_no_refs(volley.properties)
    assert pong["properties"]["ping"]["anyOf"][0] == {"type": "object", "title": "Ping"}
    # A recursive definition that declares no type is cut to "any value".
    assert volley.properties["nested"]["anyOf"][0] == {"type": "array", "items": {}}


def test_cycles_of_many_models_spell_out_each_model_once():
    # Each node type nests every other one. Inlining every path through them
    # took 37 s and 2.5 GB for seven node types.
    @tool(description="Evaluate an expression")
    def evaluate(expr: Expr) -> str:
        return "ok"

    class Evaluate(Tactic):
        name = "evaluate"
        input_model = BinOp

        def call(self, task):
            return task.op

    as_kwargs = tactic_as_function(Evaluate(), parameter_mode="kwargs")
    in_tool = Counter(spelled_out(evaluate.properties))
    in_tactic = Counter(spelled_out(as_kwargs.properties))
    binop = evaluate.properties["expr"]["anyOf"][2]

    assert_no_refs(evaluate.properties)
    assert_no_refs(as_kwargs.properties)
    assert len(json.dumps(evaluate.properties)) < 10_000
    assert [in_tool[name] for name in EXPR_NODES] == [1, 1, 1, 1, 1]
    # The input model itself becomes the parameters.
    assert [in_tactic[name] for name in EXPR_NODES] == [0, 1, 1, 1, 1]
    # Other references into the cycle name the model they stand for.
    assert binop["properties"]["left"]["anyOf"][2:] == [
        {"type": "object", "title": name} for name in EXPR_NODES
    ]
    call = as_kwargs(
        FunctionCall(
            name="evaluate",
            arguments={"op": "+", "left": {"value": 1}, "right": {"name": "x"}},
        )
    )
    assert call.success and call.result == "+"


def test_recursive_models_are_spelled_out_at_each_use_outside_their_cycle():
    class Diff(BaseModel):
        before: Tree
        after: Tree

    @tool(description="Compare two trees")
    def compare(diff: Diff) -> str:
        return "ok"

    after = compare.properties["diff"]["properties"]["after"]

    assert spelled_out(compare.properties) == ["Tree", "Tree"]
    assert after["properties"]["children"]["items"] == {"type": "object", "title": "Tree"}


def test_models_used_twice_without_recursion_are_still_inlined_in_full():
    class Address(BaseModel):
        street: str

    class Order(BaseModel):
        billing: Address
        shipping: Address = Field(description="Where to ship")

    @tool(description="Place an order")
    def place(order: Order) -> str:
        return "ok"

    order = place.properties["order"]["properties"]

    assert order["billing"]["properties"]["street"]["type"] == "string"
    assert order["shipping"]["properties"]["street"]["type"] == "string"
    assert order["shipping"]["description"] == "Where to ship"


def test_tactic_tools_with_recursive_inputs_publish_valid_non_strict_schemas():
    class Walk(Tactic):
        name = "walk"
        input_model = Tree

        def call(self, task):
            return task.value + len(task.children)

    as_task = tactic_as_function(Walk())
    as_kwargs = tactic_as_function(Walk(), parameter_mode="kwargs")

    assert_no_refs(as_task.properties)
    assert_no_refs(as_kwargs.properties)
    assert as_kwargs.properties["children"]["items"] == {"type": "object", "title": "Tree"}
    # Every field is required, but the cut-off object is not strict-safe.
    assert as_kwargs.required == ["value", "children"]
    assert as_task.strict is False and as_kwargs.strict is False
    call = as_kwargs(
        FunctionCall(name="walk", arguments={"value": 1, "children": [{"value": 2, "children": []}]})
    )
    assert call.success and call.result == 2


def test_object_parameters_are_never_strict_automatically():
    @tool(description="Store")
    def store(data: dict, tags: list[str]) -> str:
        return "ok"

    # Strict providers reject objects without ``additionalProperties: false``.
    assert store.properties["data"]["type"] == "object"
    assert store.strict is False


# -- 8: skill catalogs and copied prompts -------------------------------------


class EchoInvoker:
    def call(self, dialog, model, model_args=None, **kwargs):
        return InvokeResult(message=Message(role=Role.ASSISTANT, content="ok", name="assistant"))


@pytest.mark.parametrize(
    ("system_prompt", "prompt_args", "first_line"),
    [
        ("You are helpful.", {}, "You are helpful."),
        ("You are {role}.", {"role": "a judge"}, "You are a judge."),
    ],
)
def test_skill_descriptions_with_braces_render_literally(
    tmp_path, monkeypatch, system_prompt, prompt_args, first_line
):
    write(
        tmp_path,
        {
            ".agents/skills/json-out/SKILL.md": """
                ---
                name: json-out
                description: 'Answer as a JSON object like {"answer": "...", "confidence": 0.9}.'
                ---
                Always emit a single JSON object.
            """,
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    spec = AgentSpec.from_config(
        "assistant",
        {"model_name": "fake", "system_prompt": system_prompt, "skills": ["json-out"]},
    )
    agent = spec.build(Runtime(), EchoInvoker())

    agent.open("chat", prompt_args=prompt_args)
    agent.receive("hi")
    system = agent.current_dialog.messages[0].content

    assert agent.respond().content == "ok"
    assert system.startswith(first_line)
    assert '{"answer": "...", "confidence": 0.9}' in system


def test_copied_prompts_report_their_current_template_variables():
    prompt = Prompt(path="p", prompt="You are {role}.")
    copied = prompt.model_copy(update={"prompt": "Now {topic}."})

    assert copied.template_vars == {"topic"}
    assert copied(topic="tides") == "Now tides."
    with pytest.raises(ValueError, match="Missing required template variables"):
        copied(role="judge")
    with pytest.raises(ValueError, match="Single '\\{'"):
        Prompt(path="bad", prompt="Reply {")


# -- 12: Dialog/Prompt.from_dict never imports --------------------------------


class Verdict(BaseModel):
    ok: bool


def test_dialog_from_dict_never_imports_modules_named_in_data(tmp_path, monkeypatch):
    marker = tmp_path / "imported"
    write(
        tmp_path,
        {
            "fix2_side_effect_module.py": f"""
                import pathlib

                pathlib.Path({str(marker)!r}).write_text("imported")


                class Model:
                    pass
            """,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    saved = {
        "messages": [],
        "owner": "a",
        "session_name": "s",
        "top_prompt": {
            "path": "p",
            "prompt": "hi",
            "format": {
                "model": "fix2_side_effect_module:Model",
                "schema": {"type": "object"},
            },
        },
    }

    restored = Dialog.from_dict(json.loads(json.dumps(saved)))

    assert not marker.exists()
    assert "fix2_side_effect_module" not in sys.modules
    assert restored.top_prompt.format == {"type": "object"}


def test_from_dict_resolves_only_already_loaded_model_classes():
    prompt = Prompt(path="p", prompt="Judge.", format=Verdict)
    not_a_model = {"model": "os:system", "schema": {"type": "object"}}

    assert Prompt.from_dict(prompt.to_dict()).format is Verdict
    assert Prompt.from_dict({"path": "q", "prompt": "x", "format": not_a_model}).format == {
        "type": "object"
    }


# -- 17: custom sections named like hub sections ------------------------------


def test_hub_named_sections_with_paths_are_discovered(tmp_path):
    write(
        tmp_path,
        {
            "lllm.toml": """
                [package]
                name = "toolkit"

                [schemas]
                paths = ["schemas"]

                [datasets]
                paths = ["datasets"]

                [examples]
                paths = ["examples"]
            """,
            "schemas/user.json": '{"section": "schemas"}',
            "datasets/user.json": '{"section": "datasets"}',
            "examples/user.json": '{"section": "examples"}',
        },
    )
    runtime = Runtime()

    load_package(str(tmp_path / "lllm.toml"), runtime=runtime)

    assert runtime.keys() == [
        "toolkit.datasets:user.json",
        "toolkit.examples:user.json",
        "toolkit.schemas:user.json",
    ]


def test_hub_sections_without_paths_are_not_discovered(tmp_path):
    write(
        tmp_path,
        {
            "lllm.toml": """
                [package]
                name = "toolkit"

                [schemas.user]
                path = "schemas/user.json"

                [card]
                summary = "Tools"
            """,
            "schemas/user.json": '{"type": "object"}',
            "card/notes.txt": "not a resource",
        },
    )
    runtime = Runtime()

    load_package(str(tmp_path / "lllm.toml"), runtime=runtime)

    assert runtime.keys() == []


# -- 32: tactic_as_function(parameter_mode="kwargs") without fields -----------


def test_kwargs_mode_falls_back_to_one_task_parameter_for_inputs_without_fields():
    class Upper(Tactic):
        name = "upper"
        input_type = str
        output_type = str

        def call(self, task):
            return task.upper()

    function = tactic_as_function(Upper(), parameter_mode="kwargs")
    call = function(FunctionCall(name=function.name, arguments={"task": "hello"}))

    assert function.properties == {"task": {"type": "string"}}
    assert function.required == ["task"]
    assert call.success and call.result == "HELLO"
