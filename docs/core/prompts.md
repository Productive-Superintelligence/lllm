# Prompts

A `Prompt` is the complete behavioural definition for one agent turn. It bundles
four concerns into one object:

1. **Template**: the text sent to the model, with `{variable}` placeholders.
2. **Output contract**: a `parser` (or a structured-output `format`) that defines
   what a valid reply looks like. The agent loop keeps repairing until it gets
   one, within its retry budget.
3. **Tool surface**: the `Function` and `MCP` objects the model may call during
   this turn.
4. **Handler strategy**: a `BaseHandler` that decides what to send back when a
   tool result arrives or when parsing fails.

The key design principle: **the agent is a function, and the `Prompt` is its
type signature**. The invoker reads the parser, format, and handler from
`dialog.top_prompt`, the prompt that sent the latest turn (`agent.open()` for
the system prompt, `agent.receive_prompt()` for a user turn). The turn's tools
are that prompt's `function_list` plus the agent's own tools (those on its
system prompt), unless the prompt sets `agent_tools=False`. This is what gives
you turn-level control: every turn declares exactly what it accepts and how
failure is handled.

---

## Anatomy of a prompt

```python
from lllm import Prompt, tool
from lllm.runtimes.native import DefaultTagParser


@tool(
    description="Get current weather for a city",
    prop_desc={"location": "City name, e.g. San Francisco, CA"},
)
def get_weather(location: str, units: str = "celsius") -> str:
    return f"Sunny, 22 degrees ({units}) in {location}"


weather_prompt = Prompt(
    path="weather/bot",
    prompt=(
        "You are a weather assistant.\n"
        "Answer the user's question: {question}\n"
        "Put your answer inside <answer> tags."
    ),
    parser=DefaultTagParser(xml_tags=["answer"]),
    function_list=[get_weather],
    metadata={"author": "team-weather", "version": "1.0"},
)
```

| Field | Type | Purpose |
| --- | --- | --- |
| `path` | `str` | Identifier and registry key, e.g. `runtime.get_prompt(path)`. |
| `prompt` | `str` | Template, rendered by `renderer` (default: `str.format`). |
| `metadata` | `dict` | Tracking info: version, author, experiment name. |
| `parser` | object with `parse()`, a callable, or `None` | Defines a valid reply. `None` = free-form. |
| `format` | Pydantic model class, JSON-schema dict, or `None` | Structured output (Chat Completions API). |
| `function_list` | `list[Function \| str]` | Tools for this turn. Strings are package refs resolved when the prompt runs. |
| `mcp_servers_list` | `list[MCP]` | MCP server declarations. |
| `agent_tools` | `bool` | Also offer the agent's own tools on this turn. Default `True`; `False` for a turn that must answer without them. |
| `addon_args` | `dict` | Provider features such as web search or computer use. |
| `handler` | `BaseHandler` | Exception and tool-interrupt strategy. Default: `DefaultSimpleHandler()`. |
| `renderer` | `BaseRenderer` | Template renderer. Default: `StringFormatterRenderer()`. |

Read-only helpers: `functions` (name → `Function`), `mcp_servers`,
`template_vars`, `allow_web_search`, `computer_use_config`.

---

## Rendering

Calling a prompt renders it:

```python
from lllm import Prompt

prompt = Prompt(path="greeter", prompt="Hello {name}, welcome to {place}!")

prompt(name="Alice", place="Wonderland")   # 'Hello Alice, welcome to Wonderland!'
prompt.template_vars                        # {'name', 'place'}
prompt.validate_args({"name": "Alice"})     # ['place'] (missing variables)
```

`template_vars` is read from the current template text, so a copy made with
`prompt.model_copy(update={"prompt": ...})` reports its own variables. Every
variable must be supplied; a missing one raises `ValueError` before anything is
sent. Templates are always formatted, so literal braces are written
doubled (`{{` and `}}`) and come out single, whether or not the template has
variables:

```python
from lllm import Prompt

Prompt(path="json/hint", prompt='Reply as {{"answer": ...}}')()   # 'Reply as {"answer": ...}'
```

`Dialog.put_prompt()` and `Agent.receive_prompt()` render the prompt with the
`prompt_args` you pass.

### Custom renderer

Subclass `BaseRenderer` to change how values are rendered:

```python
from lllm import Prompt
from lllm.runtimes.native import BaseRenderer


class BulletRenderer(BaseRenderer):
    """Render list values as bullet lists, everything else with str.format."""

    def render(self, prompt: str, **kwargs) -> str:
        def fmt(value):
            if isinstance(value, (list, tuple)):
                return "\n".join(f"- {item}" for item in value)
            return value

        return prompt.format(**{key: fmt(value) for key, value in kwargs.items()})


report = Prompt(
    path="report/items",
    prompt="Summarise these findings:\n{items}",
    renderer=BulletRenderer(),
)

print(report(items=["apples", "bananas", "cherries"]))
```

The renderer runs for every render, with or without variables. Required
variables are still detected with `str.format` field syntax before it runs, and
each `{name}` field must be supplied, so a renderer should keep `{name}` fields
for its variables. Engines with their own syntax (Jinja's `{% %}` blocks) do not
fit this contract.

---

## Parsing

The parser defines the **return type** of the turn. The invoker runs it on the
reply; any exception it raises becomes a parse error, and the agent loop sends a
repair prompt until the reply parses or retries run out.

### No parser: free-form

```python
prompt = Prompt(path="chat", prompt="Say hello to {name}.")
# reply.content is the text; reply.parsed == {}
```

`prompt.parse(text)` on a parser-less prompt returns `{"raw": text}`.

### Tag parser: XML and Markdown blocks

```python
from lllm import Prompt
from lllm.runtimes.native import DefaultTagParser

prompt = Prompt(
    path="analyst/report",
    prompt=(
        "Analyse the following text: {text}\n\n"
        "Respond with:\n"
        "<reasoning>your step-by-step reasoning</reasoning>\n"
        "<answer>your final answer</answer>\n"
        "End with <DONE> when complete."
    ),
    parser=DefaultTagParser(
        xml_tags=["reasoning", "answer"],   # extracted; each must appear
        signal_tags=["DONE"],               # presence check only
    ),
)

parsed = prompt.parse(
    "<reasoning>step 1 ...</reasoning><answer>The answer is 42.</answer> <DONE>"
)
# {
#     "raw": "...",
#     "xml_tags": {"reasoning": ["step 1 ..."], "answer": ["The answer is 42."]},
#     "md_tags": {},
#     "signal_tags": {"DONE": True},
# }
```

`DefaultTagParser` fields:

| Field | Behaviour |
| --- | --- |
| `xml_tags` | `<tag>...</tag>` blocks to extract. A listed tag that never appears is a parse error. |
| `md_tags` | Fenced blocks to extract (```` ```python ... ``` ````). A listed tag that never appears is a parse error. |
| `required_xml_tags` / `required_md_tags` | Also extracted, and also errors when missing. Use them when you only list required tags. |
| `signal_tags` | `True`/`False` for whether `<TAG>` occurs anywhere. |
| `parser_args` | Free-form dict for subclasses. |

Each extracted entry is a list, because a tag may occur several times. Missing
tags raise `ParseError` with a message such as
`No answer tags found, it should be provided as <answer>...</answer>`, and that
message is what the model sees in the repair prompt. The native `ParseError`
subclasses `lllm.ParseError`, so `except lllm.ParseError` catches both.

```python
parser = DefaultTagParser(md_tags=["python"])
parser.parse("Here:\n```python\nprint('hi')\n```")["md_tags"]   # {'python': ["print('hi')"]}
```

If you need optional tags, the runtime-agnostic `lllm.DefaultTagParser` has the
same fields but only enforces the `required_*` lists. Any object with a
`parse(content, **kwargs)` method works as a prompt parser.

### Structured output: Pydantic model

For JSON-mode replies use `format` instead of `parser`:

```python
from pydantic import BaseModel


class ResearchReport(BaseModel):
    title: str
    summary: str
    confidence: float
    sources: list[str]


prompt = Prompt(
    path="researcher/report",
    prompt="Research the following topic and return a structured report: {topic}",
    format=ResearchReport,
)

# reply.parsed == {"title": "...", "summary": "...", "confidence": 0.9, "sources": [...]}
# report = ResearchReport.model_validate(reply.parsed)
```

With a Pydantic class, the LiteLLM invoker sends it as `response_format`; with a
dict it requests `{"type": "json_object"}`. The reply is decoded with
`json.loads` (invalid JSON triggers a repair turn) and the `parser` is not run.
`format` is not supported with the Responses API (`api_type="response"`).

To validate the decoded object against the model *and* get repair turns with
Pydantic's error message, use a validating parser instead; see
[Composing with Pydantic](architecture.md#composing-with-pydantic).

### Custom parser

Subclass `DefaultTagParser` to add domain validation on top of tag extraction:

```python
from lllm import Prompt
from lllm.runtimes.native import DefaultTagParser, ParseError


class GraphParser(DefaultTagParser):
    """Parse an edge list in <graph> tags and reject cycles."""

    def parse(self, content: str, **kwargs) -> dict:
        parsed = super().parse(content, **kwargs)
        edges = [
            tuple(part.strip() for part in line.split("->"))
            for line in parsed["xml_tags"]["graph"][0].strip().splitlines()
            if "->" in line
        ]
        if has_cycle(edges):
            raise ParseError(
                "The graph you returned contains a cycle. "
                "Please revise so all edges point forward."
            )
        parsed["edges"] = edges
        return parsed


def has_cycle(edges) -> bool:
    graph = {}
    for a, b in edges:
        graph.setdefault(a, []).append(b)
    seen, stack = set(), set()

    def visit(node):
        if node in stack:
            return True
        if node in seen:
            return False
        seen.add(node)
        stack.add(node)
        cyclic = any(visit(nxt) for nxt in graph.get(node, []))
        stack.discard(node)
        return cyclic

    return any(visit(node) for node in list(graph))


prompt = Prompt(
    path="planner/dag",
    prompt="Build a dependency graph for: {task}. Return edges like 'a -> b' in <graph> tags.",
    parser=GraphParser(xml_tags=["graph"]),
)
```

A parser can also be a plain function. A non-dict return value is wrapped as
`{"value": ...}`, and `"raw"` is always added:

```python
def word_count(content: str, **kwargs) -> int:
    return len(content.split())


Prompt(path="wc", prompt="Write a haiku.", parser=word_count).parse("an old silent pond")
# {'value': 4, 'raw': 'an old silent pond'}
```

`parser_args` passed to `agent.respond(parser_args={...})` arrive as keyword
arguments to `parse()`.

---

## Defining tools

LLLM has one regular tool type, `Function`, and two ways to program against it:

- **Coupled declaration and implementation:** `@tool` turns a Python function
  into a `Function` that carries both schema and implementation. Pass it
  directly in `function_list`, or reference the package resource by URL.
- **Decoupled declaration and implementation:** put a bare `Function`
  declaration in the prompt, like a header file, and provide a package
  `@tool(name=...)` implementation with the same name. LLLM binds them by
  exact name when the prompt runs.

Tactic tools and proxies are built on top of this model: a tactic tool is a
generated `Function` backed by a tactic method, and a proxy exposes many
endpoints through `query_api_doc`, `run_python`, and `CALL_API`. See
[Tactics](tactic.md#tactics-as-tools) and [Proxies & Sandbox](proxy-and-sandbox.md).

### `@tool`: coupled tools

The decorator builds the JSON schema from type hints:

```python
from lllm import Prompt, tool


@tool(
    description="Search the web for current information",
    prop_desc={
        "query": "The search query string",
        "max_results": "Maximum number of results to return",
    },
)
def web_search(query: str, max_results: int = 5) -> str:
    return f"{max_results} results for {query!r}"   # call a real API here


@tool(description="Evaluate an arithmetic expression")
def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}))   # demo only


prompt = Prompt(
    path="research/agent",
    prompt="Answer the following question using your tools: {question}",
    function_list=[web_search, calculator],
)
```

`tool(description=None, prop_desc=None, *, name=None, strict=None, processor=...)`:

- Parameter schemas follow the annotations, through Pydantic: `list[str]` gives
  an array with string items, `Literal["fast", "deep"]` an enum, a `BaseModel`
  a nested object, `Optional[...]` a nullable value. Unannotated parameters are
  strings.
- Nested models are inlined, since tool schemas have no `$defs`. A recursive
  model is spelled out once; references back into it become its type and
  title, such as `{"type": "object", "title": "Node"}`.
- Parameters without defaults are `required`; defaults are added to the
  property description.
- `strict=None` (the default) turns on strict tool schemas when every
  parameter is required and flat (a string, number, boolean, enum, or a list of
  these), which is what OpenAI-style strict mode accepts. An object parameter,
  such as a model or a dict, keeps it off, because strict mode needs
  `additionalProperties: false` on every object. Pass `strict=True` or
  `strict=False` to decide yourself.
- `description` defaults to the docstring, `name` to the function name. Names
  may not contain whitespace, `/`, `:`, `\`, `;`, or `%`.
- `Function.from_callable(fn, ...)` does the same without the decorator.

```python
from typing import Literal

from lllm import tool


@tool(description="Search the archive")
def archive_search(query: str, tags: list[str], mode: Literal["fast", "deep"]) -> str:
    return f"{mode} search for {query!r} in {tags}"


archive_search.properties["mode"]   # {'enum': ['fast', 'deep'], 'type': 'string'}
archive_search.strict               # True: every parameter required, all flat
```

When the model calls a tool, the loop runs the implementation with the call's
arguments. An exception inside the tool does not crash the loop: the model
receives `Error: <message>` as the result and can recover.

```python
from lllm import FunctionCall

call = web_search(FunctionCall(name="web_search", arguments={"query": "LLLM"}))
call.result       # '5 results for ...'
call.result_str   # formatted text the model sees
```

Package-discovered tools can be referenced by URL instead of imported:

```python
prompt = Prompt(
    path="research/agent",
    prompt="Answer the following question using your tools: {question}",
    function_list=[
        "shared_pkg.tools:web_search",
        "shared_pkg.tactics:code_review",
    ],
)
```

String refs are resolved when the prompt runs. Full URLs are exact; bare names
and `pkg:name` shorthand resolve relative to the prompt's own package, then
the runtime's default namespace. A ref that matches both a tool and a tactic is
an error; use the full URL. Proxy refs are not allowed in `function_list`; put
`pkg.proxies:name` in the agent config `tools` instead.

### `Function` declarations: schema and implementation apart

Useful when the prompt file should define *what* tools exist and a tool module
defines *how* they run:

```python
# prompts/research.py: declaration, versioned with the prompt
from lllm import Function, Prompt

search_schema = Function(
    name="search",
    description="Search an internal knowledge base",
    properties={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Number of results"},
    },
    required=["query"],
)

research = Prompt(path="system", prompt="Use search when needed.", function_list=[search_schema])
```

```python
# tools/search.py: implementation
from lllm import tool


@tool(name="search", description="Search internal documents.")
def search(query: str, top_k: int = 5) -> str:
    return my_retriever.search(query, top_k=top_k)
```

When the prompt runs, `search_schema` binds to the tool registered under
`<package>.tools:search`: the declaration keeps its own schema and borrows the
implementation. The implementation's registered key must equal the declared
name, so give a decoupled tool its own module named after it (`tools/search.py`),
or register it explicitly with `runtime.register_tool("search", search,
namespace="pkg.tools")`. If no implementation matches, resolution fails instead
of silently binding the wrong callable. Calling an unbound declaration directly
raises `RuntimeError`.

### Custom result formatting

By default a tool result reaches the model as:

```
Return of calling function <name> with arguments <args>:
---
<result>
---
```

Override it per tool with a `processor(result, function_call) -> str`:

```python
def compact_processor(result, function_call) -> str:
    return str(result)


@tool(
    description="Get stock price",
    prop_desc={"ticker": "Stock ticker symbol, e.g. AAPL"},
    processor=compact_processor,
)
def get_stock_price(ticker: str) -> str:
    return f"$123.45 ({ticker})"
```

### MCP servers

```python
from lllm import MCP, Prompt

prompt = Prompt(
    path="docs/search",
    prompt="Search the docs when needed.",
    mcp_servers_list=[
        MCP(
            server_label="docs",
            server_url="https://example.com/mcp",
            require_approval="never",       # "never" | "manual" | "auto"
            allowed_tools=["search"],
        )
    ],
)
```

`MCP.to_tool()` produces the `{"type": "mcp", ...}` entry the LiteLLM invoker
sends next to function tools.

### Agent tools and `agent_tools`

A prompt's `function_list` is the tool surface specific to its turn. When an
agent runs the prompt, it adds its own tools: everything on its system prompt,
including agent-config `tools`, proxy tools (`query_api_doc`, `run_python`), and
`activate_skill`. A user-turn prompt therefore never has to repeat them, and a
tool on the turn prompt wins over an agent tool with the same name. Set
`agent_tools=False` for a turn that must be answered without the agent's tools:

```python
from lllm import Prompt

summary = Prompt(
    path="analyst/summary",
    prompt="Summarise the findings so far in three bullets.",
    agent_tools=False,       # answer from the conversation only
)
```

The handler's final prompt, sent when the tool budget runs out, has no tools
either.

---

## Handlers

When the loop meets a parse error or a tool result, it asks
`dialog.top_prompt.on_exception(session)` or `on_interrupt(session)` for the
next prompt. The `handler` field is the object that answers. All three methods
receive the active prompt and the current `AgentCallSession` and return a
`Prompt`.

### What the default handler does

| Event | Message sent to the model | Inherits tools? |
| --- | --- | --- |
| Parse error | `"Error: {error_message}. Please fix."` | Yes |
| Tool result | `"{call_results}"` | Yes |
| Tool budget exhausted | `"You are reaching the limit of tool calls. Provide the final response."` | No |

Each message is wrapped in a child prompt via `extend()`, so it keeps the
parent's parser. The messages are fields, so simple changes need no subclass:

```python
from lllm import Prompt
from lllm.runtimes.native import DefaultSimpleHandler, DefaultTagParser

strict = Prompt(
    path="analyst/strict",
    prompt="Analyse {topic}. Put your answer in <answer> tags.",
    parser=DefaultTagParser(xml_tags=["answer"]),
    handler=DefaultSimpleHandler(
        exception_msg="That reply was not valid: {error_message}\nResend it in the required format.",
        interrupt_final_msg="Stop calling tools and give your final answer now.",
    ),
)
```

### Custom handler: rule-based

```python
from lllm import Prompt
from lllm.runtimes.native import AgentCallSession, BaseHandler, DefaultTagParser


class RetryWithHintHandler(BaseHandler):
    """First failure: a gentle hint. Later failures: a strict format reminder."""

    def on_exception(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        retry = session.exception_retries_count
        if retry <= 1:
            msg = "Your response had a formatting error: {error_message}. Please try again."
        else:
            msg = (
                f"Formatting error (attempt {retry}): {{error_message}}\n"
                "You MUST wrap your answer in <answer> tags. No other format is accepted."
            )
        return prompt.extend(path=f"__{prompt.path}_exception_{retry}", prompt=msg)

    def on_interrupt(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        return prompt.extend(path=f"__{prompt.path}_interrupt", prompt="{call_results}")

    def on_interrupt_final(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        return prompt.extend(
            path=f"__{prompt.path}_interrupt_final",
            prompt=f"You have used all {session.max_interrupt_steps} tool rounds. Answer now.",
            function_list=[],
            mcp_servers_list=[],
            addon_args={},
        )


prompt = Prompt(
    path="analyst/strict",
    prompt="Analyse {topic}. Put your answer in <answer> tags.",
    parser=DefaultTagParser(xml_tags=["answer"]),
    function_list=[web_search],
    handler=RetryWithHintHandler(),
)
```

The exception prompt is rendered with `{error_message}` and the interrupt prompt
with `{call_results}`, so keep exactly those placeholders (and escape other
braces).

### Custom handler: agentic

The handler gets the full session, so it can call another agent:

```python
from lllm import Agent, Prompt
from lllm.runtimes.native import AgentCallSession, BaseHandler, DefaultSimpleHandler


class FixerHandler(BaseHandler):
    """Ask a second agent how to repair a failed reply."""

    def __init__(self, fixer: Agent):
        self.fixer = fixer
        self.fallback = DefaultSimpleHandler()

    def on_exception(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        step = max(session.exception_retries)                 # latest loop step
        errors = "\n".join(str(e) for e in session.exception_retries[step])
        alias = f"fix_{id(session)}_{session.exception_retries_count}"

        self.fixer.open(alias)
        self.fixer.receive(
            f"A reply failed validation:\n{errors}\n\n"
            f"Original instructions:\n{prompt.prompt}\n\n"
            "Write one short repair instruction for the author."
        )
        advice = self.fixer.respond().content
        self.fixer.close(alias)

        safe = advice.replace("{", "{{").replace("}", "}}")
        return prompt.extend(path=f"__{prompt.path}_fix", prompt=safe + "\n\nParser said: {error_message}")

    def on_interrupt(self, prompt, session):
        return self.fallback.on_interrupt(prompt, session)

    def on_interrupt_final(self, prompt, session):
        return self.fallback.on_interrupt_final(prompt, session)
```

---

## Prompt composition

`extend()` creates a child prompt that inherits every field, with overrides. A
new `path` is required.

```python
base = Prompt(
    path="base/analyst",
    prompt="You are a research analyst.\n\nTask: {task}",
    parser=DefaultTagParser(xml_tags=["reasoning", "answer"]),
    function_list=[web_search, calculator],
)

# Specialise the persona, keep everything else
finance_analyst = base.extend(
    path="finance/analyst",
    prompt="You are a financial research analyst.\n\nTask: {task}",
)

# Strip tools for a lightweight summary turn
summariser = base.extend(
    path="base/summariser",
    prompt="Summarise the following research: {research}",
    function_list=[],
)

# Require an extra tag
strict_analyst = base.extend(
    path="base/analyst_strict",
    parser=DefaultTagParser(xml_tags=["reasoning", "answer", "confidence"]),
)
```

`extend()` copies field values directly rather than serializing, so custom
parsers, renderers, and handler instances carry over.

---

## Provider capabilities

Provider-specific features live in `addon_args`, so new provider features never
need new `Prompt` fields:

```python
# OpenAI web search
search_prompt = Prompt(
    path="web/searcher",
    prompt="Research this question using web search: {question}",
    addon_args={"web_search": True},
)

# Computer use
cua_prompt = Prompt(
    path="browser/agent",
    prompt="Complete the following browser task: {task}",
    addon_args={
        "computer_use": {"display_width": 1280, "display_height": 800, "environment": "browser"}
    },
)

search_prompt.allow_web_search      # True
cua_prompt.computer_use_config      # {'display_width': 1280, ...}
```

The LiteLLM invoker turns these into native tools on the Responses API
(`api_type: response`): `web_search` adds `{"type": "web_search_preview"}` and
`computer_use` adds a `computer_use_preview` tool. On the default Chat
Completions API they are ignored. See [Invokers](invokers.md#6-chat-completions-vs-responses-api).

---

## Organization and discovery

Prompts are Python objects at module scope in `.py` files. `lllm.toml` names the
folders, and discovery registers every `Prompt` it finds:

```toml
# lllm.toml
[package]
name = "my_pkg"

[prompts]
paths = ["prompts"]
```

The registry key is the file path (without `.py`) plus the prompt's own `path`.
A prompt with `path="system"` in `prompts/weather/bot.py` registers as
`my_pkg.prompts:weather/bot/system`:

```
prompts/
├── weather/
│   └── bot.py          # system, analysis
├── finance/
│   └── analyst.py      # research, summary
└── coding/
    └── agent.py        # planner, executor, reviewer
```

Register without discovery:

```python
from lllm import register_prompt

register_prompt(my_prompt)          # default runtime, key = my_prompt.path, overwrites
# register_prompt(my_prompt, overwrite=False) raises ValueError if the key exists

runtime.register_prompt(my_prompt, namespace="my_pkg.prompts")   # a specific runtime
```

Retrieve:

```python
from lllm import load_prompt

prompt = load_prompt("weather/bot/system")               # default package
prompt = load_prompt("my_pkg:weather/bot/system")        # package-qualified
prompt = runtime.get_prompt("my_pkg.prompts:weather/bot/system")
```

A registered prompt remembers its package, which is what lets bare tool refs
and `Function` declarations on it resolve inside that package. See
[Package Discovery](../packages/discovery.md).

---

## Metadata and tracking

`metadata` accepts any JSON-like dict. `info_dict()` produces a snapshot for
experiment tracking:

```python
prompt = Prompt(
    path="analyst/v2",
    prompt="Analyse {topic}.",
    metadata={"author": "junyan", "experiment": "ablation-no-cot", "version": "2.1"},
)

prompt.info_dict()
# {
#     "path": "analyst/v2",
#     "prompt_hash": "...",            # first 12 hex chars of SHA-256 of the template
#     "metadata": {"author": "junyan", "experiment": "ablation-no-cot", "version": "2.1"},
#     "functions": [],
#     "mcp_servers": [],
#     "addon_args": {},
#     "has_parser": False,
#     "has_format": False,
# }
```

`prompt_hash` changes whenever the template text changes, which makes prompt
drift across experiments easy to detect.

`to_dict()` returns a JSON-safe snapshot: tools are stored as their schemas
(implementations are not included; they are re-bound by name to registered
`@tool` implementations when the prompt next runs), a Pydantic `format` model is
stored as `{"model": "module:Qualname", "schema": {...}}`, and the parser,
handler, and renderer are left out. `Prompt.from_dict()` rebuilds the prompt
without importing code: it uses the `format` model class only when its module
is already imported, and the stored JSON schema otherwise.

---

## Complete example: a prompt file

A real prompt module. Discovery registers every module-level `Prompt`, prefixed
with the file path.

```python
# prompts/research/agent.py
#
# Registered keys (package "research_pkg"):
#   research_pkg.prompts:research/agent/system
#   research_pkg.prompts:research/agent/task
#   research_pkg.prompts:research/agent/summarise

from lllm import Function, Prompt, tool
from lllm.runtimes.native import DefaultTagParser, ParseError


# -- Tools -------------------------------------------------------------------

@tool(
    description="Search the web for current information on a topic",
    prop_desc={"query": "Search query string"},
)
def web_search(query: str) -> str:
    return f"[Search results for: {query}]"   # call a real search API here


@tool(description="Fetch the full text of a URL", prop_desc={"url": "The URL to fetch"})
def fetch_url(url: str) -> str:
    return f"[Content of: {url}]"


# Declared here for the model; implemented by tools/save_note.py in this package.
save_note = Function(
    name="save_note",
    description="Save a research note to the session store",
    properties={
        "title": {"type": "string", "description": "Note title"},
        "content": {"type": "string", "description": "Note content"},
    },
    required=["title", "content"],
)


# -- Parser ------------------------------------------------------------------

class ResearchParser(DefaultTagParser):
    """Tag extraction plus a range check on <confidence>."""

    def parse(self, content: str, **kwargs) -> dict:
        parsed = super().parse(content, **kwargs)
        raw = parsed["xml_tags"]["confidence"][0].strip()
        try:
            confidence = float(raw)
        except ValueError:
            raise ParseError(f"<confidence> must be a number, got {raw!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ParseError(f"<confidence> must be between 0 and 1, got {confidence}")
        parsed["confidence"] = confidence
        return parsed


# -- Prompts -----------------------------------------------------------------

system = Prompt(
    path="system",
    prompt=(
        "You are a careful research assistant with access to web search.\n"
        "Verify claims with at least two sources before concluding.\n"
        "Current date: {date}"
    ),
    metadata={"role": "system"},
)

task = Prompt(
    path="task",
    prompt=(
        "Research the following question thoroughly: {question}\n\n"
        "Use your tools, then respond with:\n"
        "<reasoning>your research process</reasoning>\n"
        "<answer>your final, sourced answer</answer>\n"
        "<confidence>a number between 0 and 1</confidence>"
    ),
    parser=ResearchParser(xml_tags=["reasoning", "answer", "confidence"]),
    function_list=[web_search, fetch_url, save_note],
    metadata={"version": "1.2"},
)

summarise = task.extend(
    path="summarise",
    prompt=(
        "Summarise these findings in 2-3 sentences: {findings}\n\n"
        "<answer>your summary</answer>"
    ),
    function_list=[],
    parser=DefaultTagParser(xml_tags=["answer"]),
)
```

```python
# tools/save_note.py
from lllm import tool

NOTES: list[tuple[str, str]] = []


@tool(name="save_note", description="Save a research note to the session store")
def save_note(title: str, content: str) -> str:
    NOTES.append((title, content))
    return f"Note {title!r} saved."
```

And a tactic that uses these prompts:

```python
# tactics/researcher.py
import datetime

from lllm import Tactic


class Researcher(Tactic):
    name = "researcher"
    agent_group = ["researcher"]

    def call(self, task: str) -> str:
        agent = self.agents["researcher"]
        agent.open("research", prompt_args={"date": datetime.date.today().isoformat()})

        agent.receive_prompt(agent.runtime.get_prompt("research/agent/task"), {"question": task})
        response = agent.respond()
        answer = response.parsed["xml_tags"]["answer"][0]

        if response.parsed["confidence"] < 0.5:
            # Low confidence: consolidate in a tool-free summary turn.
            agent.receive_prompt(
                agent.runtime.get_prompt("research/agent/summarise"), {"findings": answer}
            )
            answer = agent.respond().parsed["xml_tags"]["answer"][0]
        return answer
```

```yaml
# configs/default.yaml
tactic_type: researcher
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.2
agent_configs:
  - name: researcher
    system_prompt_path: research/agent/system
```

```python
from lllm import build_tactic, load_runtime, resolve_config

load_runtime("research_pkg/lllm.toml")
tactic = build_tactic(resolve_config("default"))
print(tactic("What architecture do modern LLMs use?"))
```
