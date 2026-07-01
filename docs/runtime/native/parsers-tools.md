# Parsers And Tools

Native parsers and tools are part of the agent loop. Parsers turn model text
into structured data; handlers decide how repair and tool-result prompts are
fed back into the next model call.

## Tagged Parsing

`DefaultTagParser` understands XML blocks, fenced markdown blocks, and signal
tags.

```python
from lllm.runtimes.native import DefaultTagParser, Prompt

parser = DefaultTagParser(
    required_xml_tags=["answer"],
    md_tags=["python"],
    signal_tags=["DONE"],
)

prompt = Prompt(
    path="analysis/tagged",
    prompt="Return <answer>...</answer>. Put code in ```python blocks.",
    parser=parser,
)

parsed = prompt.parse(
    "<answer>Use the native runtime.</answer>\n"
    "```python\nprint('ok')\n```\n"
    "<DONE>"
)

assert parsed["xml_tags"]["answer"] == ["Use the native runtime."]
assert parsed["signal_tags"]["DONE"] is True
```

If required blocks are missing, `DefaultTagParser` raises `ParseError`. The
agent records that failure on `AgentCallSession`, appends the prompt returned
by `prompt.on_exception(session)`, and recalls the model until the configured
exception cap is reached.

## Handlers

`DefaultSimpleHandler` creates:

- an exception prompt with `Error: {error_message}. Please fix.`;
- a tool-result prompt with `{call_results}`;
- a final prompt when the tool-call limit is reached.

Replace `handler` when parser repair or tool-result feedback needs richer
state. A handler can inspect `AgentCallSession` and return full `Prompt`
objects, not just strings.

## Function Tools

A native `Function` has two halves:

- a schema shown to the model;
- an optional Python implementation called when the model emits a tool call.

```python
from lllm.runtimes.native import FunctionCall, Prompt, tool


@tool(
    description="Add two values.",
    prop_desc={"left": "Left value.", "right": "Right value."},
)
def add(left: int, right: int = 1) -> int:
    return left + right


prompt = Prompt(
    path="math/solve",
    prompt="Solve {question}. Use tools if helpful.",
    function_list=[add],
)

call = add(FunctionCall(name="add", arguments={"left": 2, "right": 3}))
assert call.result == 5
```

`Function.from_callable()` inspects signatures and type hints. Tool names are
validated so they can be used safely in provider payloads and resource refs.
`Function.to_tool()` returns a copy of a LiteLLM/OpenAI-compatible tool schema,
so mutating the returned dictionary does not mutate the original prompt.

## Tool References

String entries in `function_list` are resolved at call time:

```python
prompt = Prompt(
    path="review/main",
    prompt="Review this patch.",
    function_list=[
        "shared.tools:search_code",
        "shared.tactics:lint_patch",
    ],
)
```

Regular tool refs bind to registered `Function` resources. Tactic refs are
wrapped as native `Function` tools with `tactic_as_function()`.

## MCP Servers

Native prompts can declare MCP servers:

```python
from lllm.runtimes.native import MCP, Prompt

prompt = Prompt(
    path="docs/search",
    prompt="Search the docs when needed.",
    mcp_servers_list=[
        MCP(
            server_label="docs",
            server_url="https://example.invalid/mcp",
            require_approval="manual",
            allowed_tools=["search"],
        )
    ],
)
```

`MCP.to_tool()` converts the declaration for LiteLLM. Approval mode is
validated as `never`, `manual`, or `auto`.
