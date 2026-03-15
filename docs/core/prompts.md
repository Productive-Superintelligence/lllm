# Prompts

A `Prompt` is the complete behavioural definition for one agent turn. It bundles four concerns into a single object:

1. **Template** — the text sent to the LLM, with `{variable}` placeholders.
2. **Output contract** — a `parser` (and optionally a `format`) that defines what a valid response looks like. The agent loop retries until it gets one.
3. **Tool surface** — the `Function` and `MCP` objects the LLM can call during this turn.
4. **Handler strategy** — a `BaseHandler` that decides what prompt to send back when a tool result arrives or when parsing fails.

The key design principle: **the agent is a function, the `Prompt` is its type signature**. The invoker reads everything it needs — tools, format, parser — from `dialog.top_prompt`. Whatever prompt was last sent via `send_message` becomes the active specification for that turn.

---

## Anatomy of a Prompt

```python
from lllm.core.models import Prompt, DefaultTagParser, tool

@tool(
    description="Get current weather for a city",
    prop_desc={"location": "City name, e.g. San Francisco, CA"},
)
def get_weather(location: str, units: str = "celsius") -> str:
    return f"Sunny, 22°C in {location}"


weather_prompt = Prompt(
    path="weather/bot",
    prompt=(
        "You are a weather assistant.\n"
        "Answer the user's question: {question}\n"
        "Put your answer inside <answer> tags."
    ),
    parser=DefaultTagParser(
        xml_tags=["answer"],
        required_xml_tags=["answer"],
    ),
    function_list=[get_weather],
    metadata={"author": "team-weather", "version": "1.0"},
)
```

**Fields at a glance:**

| Field | Type | Purpose |
|---|---|---|
| `path` | `str` | Unique identifier in the registry. Used by `runtime.get_prompt(path)`. |
| `prompt` | `str` | Template string. Rendered via `renderer` (default: `str.format`). |
| `parser` | `BaseParser \| None` | Defines valid output shape. `None` = raw passthrough. |
| `format` | `type \| dict \| None` | Pydantic model or JSON schema for structured output. |
| `function_list` | `list[Function]` | Tools the LLM can call this turn. |
| `mcp_servers_list` | `list[MCP]` | MCP server descriptors. |
| `addon_args` | `dict` | Provider-specific capabilities (web search, computer use, etc.). |
| `handler` | `BaseHandler` | Handles exceptions and tool interrupts. Default: `DefaultHandler`. |
| `renderer` | `BaseRenderer` | Renders the template. Default: `StringFormatterRenderer`. |
| `metadata` | `dict` | Arbitrary tracking info (version, author, experiment name, etc.). |

---

## Rendering

Calling a `Prompt` renders its template:

```python
prompt = Prompt(
    path="greeter",
    prompt="Hello {name}, welcome to {place}!",
)

rendered = prompt(name="Alice", place="Wonderland")
# "Hello {name}, welcome to {place}!" → "Hello Alice, welcome to Wonderland!"
```

With no arguments, the raw template string is returned unchanged. Literal braces in the template must be doubled to escape them: `{{` and `}}`.

### Custom renderer

Swap in any template engine by subclassing `BaseRenderer`:

```python
from lllm.core.models import BaseRenderer

class JinjaRenderer(BaseRenderer):
    def __init__(self):
        from jinja2 import Environment
        self.env = Environment()

    def render(self, prompt: str, **kwargs) -> str:
        return self.env.from_string(prompt).render(**kwargs)


prompt = Prompt(
    path="jinja/report",
    prompt="{% for item in items %}- {{ item }}\n{% endfor %}",
    renderer=JinjaRenderer(),
)

rendered = prompt(items=["apples", "bananas", "cherries"])
# "- apples\n- bananas\n- cherries\n"
```

The framework takes no dependency on Jinja or any template engine — `BaseRenderer` is just the extension point.

---

## Parsing

The parser defines the **return type** of the agent turn. The agent loop retries until the LLM produces output that parses successfully, or until retries are exhausted.

### No parser — raw passthrough

```python
prompt = Prompt(path="chat", prompt="Say hello to {name}.")

# response.parsed == {"raw": "Hello, Alice!"}
```

### Tag parser — structured XML / Markdown blocks

```python
from lllm.core.models import DefaultTagParser

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
        xml_tags=["reasoning", "answer"],   # extract these if present
        required_xml_tags=["answer"],        # fail parse if missing
        signal_tags=["DONE"],                # boolean presence check
    ),
)

# On success, response.parsed looks like:
# {
#     "raw": "...",
#     "xml_tags": {
#         "reasoning": ["step 1: ...\nstep 2: ..."],
#         "answer":    ["The answer is 42."],
#     },
#     "md_tags": {},
#     "signal_tags": {"DONE": True},
# }
```

If `<answer>` is missing, `DefaultTagParser` raises `ParseError`, the agent loop catches it, and the exception handler kicks in automatically.

Markdown code blocks are extracted similarly via `md_tags`:

```python
parser=DefaultTagParser(
    md_tags=["python", "json"],
    required_md_tags=["python"],
)
# Extracts ```python ... ``` blocks from the response.
```

### Structured output — Pydantic model

For JSON-mode responses use `format` instead of `parser`:

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

# response.parsed == {"title": "...", "summary": "...", "confidence": 0.9, "sources": [...]}
# Access as: ResearchReport(**response.parsed)
```

Note: `format` is not supported with the Responses API (`api_type="response"`). Use the default completion API for structured output.

### Custom parser

Subclass `DefaultTagParser` to add domain-specific validation on top of standard tag extraction:

```python
from lllm.core.models import DefaultTagParser
from lllm.core.const import ParseError

class GraphParser(DefaultTagParser):
    """Parses an agent response that includes a graph definition, and validates it has no cycles."""

    def parse(self, content: str, **kwargs) -> dict:
        parsed = super().parse(content, **kwargs)
        raw_nodes = parsed["xml_tags"].get("graph", [""])[0]
        graph = build_graph(raw_nodes)
        if has_cycle(graph):
            raise ParseError(
                "The graph you returned contains a cycle. "
                "Please revise so all edges point forward."
            )
        parsed["graph"] = graph
        return parsed


prompt = Prompt(
    path="planner/dag",
    prompt="Build a dependency graph for: {task}. Return it in <graph> tags.",
    parser=GraphParser(
        xml_tags=["graph", "reasoning"],
        required_xml_tags=["graph"],
    ),
)
```

---

## Defining Tools

### `@tool` decorator — schema and implementation together

The common case. The decorator inspects type hints and builds the JSON schema automatically:

```python
from lllm.core.models import tool

@tool(
    description="Search the web for current information",
    prop_desc={
        "query": "The search query string",
        "max_results": "Maximum number of results to return (default 5)",
    },
)
def web_search(query: str, max_results: int = 5) -> str:
    # Your implementation here
    results = ...
    return results


@tool(description="Execute a Python expression and return the result")
def calculator(expression: str) -> str:
    return str(eval(expression))


prompt = Prompt(
    path="research/agent",
    prompt="Answer the following question using your tools: {question}",
    function_list=[web_search, calculator],
)
```

### `Function` + `link_function` — schema and implementation apart

Useful when the prompt file defines *what* tools exist (for the LLM to see) and the implementation is wired up separately at runtime — for example, from a proxy or a mock in tests:

```python
from lllm.core.models import Function

# Schema defined in the prompt file — versionable, readable
search_schema = Function(
    name="search",
    description="Search an internal knowledge base",
    properties={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Number of results"},
    },
    required=["query"],
)

# Implementation linked at runtime
search_schema.link_function(my_retriever.search)

# Or check if it's linked before calling
assert search_schema.linked, "search tool has no implementation"
```

### Custom result formatting

By default, tool results are formatted as:

```
Return of calling function <name> with arguments <args>:
---
<result>
---
```

Override this per-tool with a custom processor:

```python
def compact_processor(result, function_call) -> str:
    # Return only the result itself, no boilerplate
    return str(result)


@tool(
    description="Get stock price",
    prop_desc={"ticker": "Stock ticker symbol, e.g. AAPL"},
    processor=compact_processor,
)
def get_stock_price(ticker: str) -> str:
    return f"${fetch_price(ticker):.2f}"
```

---

## Handlers

When the agent loop encounters a tool result or a parse error, it calls `dialog.top_prompt.on_exception(call_state)` or `on_interrupt(call_state)` to get the next prompt to send. The `handler` field on `Prompt` is the object that answers that question.

The default handler (`DefaultHandler`) has built-in sensible behaviour. Swap it by subclassing `BaseHandler` when you need custom logic.

### What the default handler does

| Event | Message sent to LLM | Inherits tools? |
|---|---|---|
| Parse error / exception | `"Error: {error_message}. Please fix."` | Yes |
| Tool result (interrupt) | `"{call_results}"` | Yes |
| Interrupt budget exhausted | `"You are reaching the limit of tool calls. Provide the final response."` | No |

The message is wrapped into a lightweight child `Prompt` via `extend()`, so it inherits the parent's parser and other settings.

### Custom handler — rule-based

```python
from lllm.core.models import BaseHandler, Prompt, AgentCallState

class RetryWithHintHandler(BaseHandler):
    """
    On first exception: give a gentle hint.
    On subsequent exceptions: give a strict format reminder with the schema.
    """

    def on_exception(self, prompt: Prompt, call_state: AgentCallState) -> Prompt:
        retry_count = call_state.exception_retries_count
        if retry_count <= 1:
            msg = "Your response had a formatting error: {error_message}. Please try again."
        else:
            msg = (
                "Formatting error (attempt {retry_count}): {{error_message}}.\n"
                "You MUST wrap your answer in <answer> tags. No other format is accepted."
            ).format(retry_count=retry_count)
        return prompt.extend(
            path=f"__{prompt.path}_exception_{retry_count}",
            prompt=msg,
            function_list=prompt.function_list,
            mcp_servers_list=prompt.mcp_servers_list,
            addon_args=prompt.addon_args,
        )

    def on_interrupt(self, prompt: Prompt, call_state: AgentCallState) -> Prompt:
        # Standard tool-result passthrough
        return prompt.extend(
            path=f"__{prompt.path}_interrupt",
            prompt="{call_results}",
            function_list=prompt.function_list,
            mcp_servers_list=prompt.mcp_servers_list,
            addon_args=prompt.addon_args,
        )

    def on_interrupt_final(self, prompt: Prompt, call_state: AgentCallState) -> Prompt:
        steps = call_state.max_interrupt_steps
        return prompt.extend(
            path=f"__{prompt.path}_interrupt_final",
            prompt=f"You have used all {steps} tool call rounds. Now provide your final answer.",
            function_list=[],
            mcp_servers_list=[],
            addon_args={},
        )


prompt = Prompt(
    path="analyst/strict",
    prompt="Analyse {topic}. Put your answer in <answer> tags.",
    parser=DefaultTagParser(xml_tags=["answer"], required_xml_tags=["answer"]),
    function_list=[web_search],
    handler=RetryWithHintHandler(),
)
```

### Custom handler — agentic (meta-agent)

`call_state` gives you everything you need to build handlers that themselves call an LLM:

```python
class BugFixingHandler(BaseHandler):
    """Uses a second agent to diagnose and repair the broken response."""

    def __init__(self, fixer_agent):
        self.fixer_agent = fixer_agent

    def on_exception(self, prompt: Prompt, call_state: AgentCallState) -> Prompt:
        # call_state.exception_retries tells you what went wrong at each step
        last_errors = list(call_state.exception_retries.values())[-1]
        diagnosis = self.fixer_agent.diagnose(last_errors, prompt)
        return prompt.extend(
            path=f"__{prompt.path}_bugfix",
            prompt=diagnosis,
            function_list=prompt.function_list,
            mcp_servers_list=prompt.mcp_servers_list,
            addon_args=prompt.addon_args,
        )

    def on_interrupt(self, prompt, call_state):
        # Fall back to default interrupt behaviour
        return prompt.extend(
            path=f"__{prompt.path}_interrupt",
            prompt="{call_results}",
            function_list=prompt.function_list,
            mcp_servers_list=prompt.mcp_servers_list,
            addon_args=prompt.addon_args,
        )

    def on_interrupt_final(self, prompt, call_state):
        return prompt.extend(
            path=f"__{prompt.path}_interrupt_final",
            prompt="Provide your final answer now.",
            function_list=[],
            mcp_servers_list=[],
            addon_args={},
        )
```

---

## Prompt Composition

`extend()` creates a child prompt inheriting all fields, with specified overrides. A new `path` is always required.

```python
base = Prompt(
    path="base/analyst",
    prompt="You are a research analyst.\n\nTask: {task}",
    parser=DefaultTagParser(
        xml_tags=["reasoning", "answer"],
        required_xml_tags=["answer"],
    ),
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

# Add a stricter parser
strict_analyst = base.extend(
    path="base/analyst_strict",
    parser=DefaultTagParser(
        xml_tags=["reasoning", "answer", "confidence"],
        required_xml_tags=["reasoning", "answer", "confidence"],
    ),
)
```

Because `extend()` builds from field values directly (not serialization), non-serializable objects like custom parsers, renderers, and handler instances are copied correctly.

---

## Provider Capabilities

Provider-specific features live in `addon_args` rather than dedicated fields. This means new provider features never require changes to `Prompt`.

```python
# OpenAI web search (Responses API only)
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
        "computer_use": {
            "display_width": 1280,
            "display_height": 800,
            "environment": "browser",
        }
    },
)

# Convenience read-only properties
search_prompt.allow_web_search      # True
cua_prompt.computer_use_config      # {"display_width": 1280, ...}
```

The invoker reads `addon_args` and translates entries to provider-specific tool configurations. The `Prompt` itself stays provider-agnostic.

---

## Organization and Discovery

Prompts live as Python objects at module scope in `.py` files. `lllm.toml` designates one or more folders, and auto-discovery registers every `Prompt` object it finds.

```toml
# lllm.toml
[prompts]
folders = ["prompts/"]
```

Folder structure maps to path prefixes. A prompt with `path="system"` in `prompts/weather/bot.py` is registered as `weather/bot/system`:

```
prompts/
├── weather/
│   └── bot.py          # system, analysis_prompt
├── finance/
│   └── analyst.py      # research_prompt, summary_prompt
└── coding/
    └── agent.py        # planner, executor, reviewer
```

Register manually without discovery:

```python
from lllm.core.models import register_prompt

register_prompt(my_prompt)              # overwrites by default
register_prompt(my_prompt, overwrite=False)  # raises if path already registered
```

Retrieve from the runtime:

```python
from lllm.core.runtime import get_default_runtime

runtime = get_default_runtime()
prompt = runtime.get_prompt("weather/bot/system")
```

---

## Metadata and Tracking

`metadata` accepts any JSON-serializable dict. `info_dict()` produces a snapshot suitable for experiment tracking systems:

```python
prompt = Prompt(
    path="analyst/v2",
    prompt="...",
    metadata={"author": "junyan", "experiment": "ablation-no-cot", "version": "2.1"},
)

prompt.info_dict()
# {
#     "path": "analyst/v2",
#     "prompt_hash": "a1b2c3d4e5f6",   # first 12 chars of SHA-256 of the template
#     "metadata": {"author": "junyan", "experiment": "ablation-no-cot", "version": "2.1"},
#     "functions": ["web_search", "calculator"],
#     "mcp_servers": [],
#     "addon_args": {},
#     "has_parser": True,
#     "has_format": False,
# }
```

The `prompt_hash` changes whenever the template text changes, making it easy to detect prompt drift across experiments.

---

## Complete Example — a prompt file

Below is what a real prompt `.py` file looks like. Discovery will register all module-level `Prompt` objects automatically, prefixed with the folder/file namespace.

```python
# prompts/research/agent.py
#
# Registered paths (after discovery):
#   research/agent/system
#   research/agent/task
#   research/agent/summarise

from lllm.core.models import Prompt, DefaultTagParser, tool, Function
from lllm.core.const import ParseError


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(
    description="Search the web for current information on a topic",
    prop_desc={"query": "Search query string"},
)
def web_search(query: str) -> str:
    # Real implementation would call an API
    return f"[Search results for: {query}]"


@tool(
    description="Fetch the full text of a URL",
    prop_desc={"url": "The URL to fetch"},
)
def fetch_url(url: str) -> str:
    return f"[Content of: {url}]"


# Declared here for the LLM to see; implementation linked at runtime from a proxy
save_note = Function(
    name="save_note",
    description="Save a research note to the session store",
    properties={
        "title": {"type": "string", "description": "Note title"},
        "content": {"type": "string", "description": "Note content"},
    },
    required=["title", "content"],
)


# ---------------------------------------------------------------------------
# Custom parser
# ---------------------------------------------------------------------------

class ResearchParser(DefaultTagParser):
    """
    Extends tag extraction with a confidence-score range check.
    The agent must provide a confidence value between 0.0 and 1.0.
    """

    def parse(self, content: str, **kwargs) -> dict:
        parsed = super().parse(content, **kwargs)
        raw_conf = parsed["xml_tags"].get("confidence", [""])[0].strip()
        try:
            conf = float(raw_conf)
        except ValueError:
            raise ParseError(
                f"<confidence> must be a float, got: {raw_conf!r}"
            )
        if not 0.0 <= conf <= 1.0:
            raise ParseError(
                f"<confidence> must be between 0 and 1, got {conf}"
            )
        parsed["confidence"] = conf
        return parsed


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# System-level persona — loaded once by Agent.init_dialog
system = Prompt(
    path="system",
    prompt=(
        "You are a careful research assistant with access to web search.\n"
        "Always verify claims with at least two sources before concluding.\n"
        "Current date: {date}"
    ),
    metadata={"role": "system"},
)

# Main research task — sent per-request via Agent.send_message
task = Prompt(
    path="task",
    prompt=(
        "Research the following question thoroughly: {question}\n\n"
        "Use your tools to gather information, then respond with:\n"
        "<reasoning>your step-by-step research process</reasoning>\n"
        "<answer>your final, sourced answer</answer>\n"
        "<confidence>a float between 0 and 1</confidence>"
    ),
    parser=ResearchParser(
        xml_tags=["reasoning", "answer", "confidence"],
        required_xml_tags=["answer", "confidence"],
    ),
    function_list=[web_search, fetch_url, save_note],
    metadata={"version": "1.2", "author": "research-team"},
)

# Summary turn — no tools, just synthesis
summarise = task.extend(
    path="summarise",
    prompt=(
        "Summarise the following research findings in 2-3 sentences: {findings}\n\n"
        "<answer>your summary</answer>"
    ),
    function_list=[],   # no tools needed
    parser=DefaultTagParser(
        xml_tags=["answer"],
        required_xml_tags=["answer"],
    ),
)
```

And how an `Orchestra` uses these prompts:

```python
# agents/researcher.py

from lllm.core.agent import Orchestra
from lllm.core.runtime import get_default_runtime
import datetime


class ResearchAgent(Orchestra):
    agent_type = "researcher"
    agent_group = ["researcher"]

    def call(self, task: str, **kwargs) -> str:
        runtime = get_default_runtime()

        agent = self.agents["researcher"]
        dialog = agent.init_dialog(
            prompt_args={"date": datetime.date.today().isoformat()}
        )

        # Link the runtime implementation to the declared schema
        task_prompt = runtime.get_prompt("research/agent/task")
        task_prompt.link_function("save_note", self._save_note)

        agent.send_message(dialog, task_prompt, prompt_args={"question": task})
        response, dialog, call_state = agent.call(dialog)

        answer = response.parsed["xml_tags"]["answer"][0]
        confidence = response.parsed["confidence"]

        if confidence < 0.5:
            # Low confidence: run a summary turn to consolidate findings
            summarise_prompt = runtime.get_prompt("research/agent/summarise")
            agent.send_message(
                dialog, summarise_prompt, prompt_args={"findings": answer}
            )
            response, dialog, call_state = agent.call(dialog)
            answer = response.parsed["xml_tags"]["answer"][0]

        return answer

    def _save_note(self, title: str, content: str) -> str:
        # Real implementation would persist to a store
        print(f"[Note saved] {title}")
        return f"Note '{title}' saved successfully."
```