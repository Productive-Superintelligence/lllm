# Prompts

In LLLM, a Prompt is a complete behaviour definition for one agent turn. It bundles four concerns into a single object: the template text, the output contract (how to parse the response), the tool surface (what functions are available), and the handler strategy (how to recover from errors and feed tool results back).

This design follows the framework's **Functional Design** principle: the agent is a function, the parser defines the return type, and the agent call loop enforces it. A Prompt is essentially the type signature of that function.

## Anatomy of a Prompt

```python
from lllm import Prompt, Function, tool
from lllm.core.models import DefaultTagParser

@tool(
    description="Get current weather for a city",
    prop_desc={"location": "City name, e.g. San Francisco"},
)
def get_weather(location: str, units: str = "celsius") -> str:
    return "Sunny, 22°C"

weather_prompt = Prompt(
    path="weather/bot",
    prompt="You are a weather bot. The user asks: {question}",
    parser=DefaultTagParser(
        xml_tags=["answer"],
        required_xml_tags=["answer"],
    ),
    function_list=[get_weather],
    on_exception="Your response was malformed: {error_message}. Try again.",
)
```

The four groups of fields:

**Template** — `path` identifies the prompt in the registry. `prompt` is the template string with `{variable}` placeholders. `renderer` controls how placeholders are filled (default: Python `str.format`).

**Output contract** — `parser` is an instance of `BaseParser` (or `None` for raw passthrough). `format` optionally specifies a Pydantic model or JSON schema for structured output. The parser defines the "return type" — the agent loop retries until parsing succeeds or retries are exhausted.

**Tools** — `function_list` declares which tools the LLM can call this turn. `mcp_servers_list` declares MCP tool servers. `addon_args` holds provider-specific capabilities like `{"web_search": True}`.

**Handlers** — `on_exception`, `on_interrupt`, and `on_interrupt_final` define how the agent loop recovers from errors and feeds tool results back. Each can be a simple template string or a full custom Prompt.

## Organization and Discovery

Prompts live as Python objects in `.py` files. The `lllm.toml` config designates one or more prompt folders, and auto-discovery registers every `Prompt` object found at module scope.

Folder structure maps to path namespaces:

```
prompts/
  ├── weather/
  │   └── bot.py          # contains: system_prompt, analysis_prompt
  └── finance/
      └── analyst.py       # contains: research_prompt
```

A prompt defined as `Prompt(path="system", ...)` in `prompts/weather/bot.py` is registered as `weather/bot/system`. This folder acts as a prompt database — organized by domain, searchable by path.

Prompts can also be registered manually without discovery:

```python
from lllm import register_prompt

register_prompt(my_prompt)
```

## Defining Tools

LLLM provides two ways to define tools depending on whether the schema and implementation live together or apart.

**Together — the `@tool` decorator** (common case):

```python
from lllm import tool

@tool(
    description="Search the web for a query",
    prop_desc={
        "query": "The search query string",
        "max_results": "Maximum number of results to return",
    },
)
def web_search(query: str, max_results: int = 10) -> str:
    # implementation here
    return results
```

The decorator inspects type hints and builds the JSON schema automatically. The result is a `Function` instance with the callable already linked — drop it directly into `function_list`.

**Apart — `Function` + `link_function`** (proxy / separate-definition case):

```python
from lllm import Function

# Schema defined in prompt file (declarative, versionable)
search_tool = Function(
    name="search",
    description="Search the web",
    properties={
        "query": {"type": "string", "description": "Search query"},
    },
    required=["query"],
)

# Implementation linked at runtime (e.g., from a proxy)
search_tool.link_function(proxy.search)
```

This separation is useful when prompts define *what* tools exist (for the LLM to reason about) while the actual implementation comes from proxies, mocks, or different environments.

You can also build a `Function` from a callable without the decorator:

```python
search_tool = Function.from_callable(
    my_search_function,
    description="Search the web",
    prop_desc={"query": "The search query"},
)
```

## Parsing

Parsing is the mechanism that makes agents function-like. The parser defines the expected output shape, and the agent loop retries until the output matches (or retries are exhausted).

**No parser** — the raw LLM output is returned as `{"raw": content}`:

```python
Prompt(path="chat", prompt="Hello {name}")
# response.parsed == {"raw": "Hi there!"}
```

**Tag parser** — extracts structured data from XML/Markdown tags:

```python
from lllm.core.models import DefaultTagParser

Prompt(
    path="analyst",
    prompt="Analyze {topic}. Put your answer in <answer> tags.",
    parser=DefaultTagParser(
        xml_tags=["answer", "reasoning"],
        required_xml_tags=["answer"],
        signal_tags=["DONE"],
    ),
)
# response.parsed == {
#     "raw": "...",
#     "xml_tags": {"answer": ["..."], "reasoning": ["..."]},
#     "md_tags": {},
#     "signal_tags": {"DONE": True},
# }
```

Missing required tags raise `ParseError`, which the agent loop catches and routes to the exception handler for retry.

**Structured output** — use a Pydantic model for JSON-validated responses:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float

Prompt(
    path="analyst",
    prompt="Analyze {topic}",
    format=Analysis,
)
```

**Custom parser** — subclass `BaseParser` for domain-specific validation:

```python
from lllm.core.models import BaseParser, DefaultTagParser

class GraphParser(DefaultTagParser):
    def parse(self, content, **kwargs):
        parsed = super().parse(content, **kwargs)
        # Custom validation: check for cycles in the output graph
        graph = build_graph(parsed["xml_tags"]["nodes"])
        if has_cycle(graph):
            raise ParseError("Output graph contains a cycle")
        parsed["graph"] = graph
        return parsed
```

## Rendering

By default, `Prompt.__call__` uses Python's `str.format`:

```python
prompt = Prompt(path="chat", prompt="Hello {name}, welcome to {place}")
rendered = prompt(name="Alice", place="Wonderland")
# "Hello Alice, welcome to Wonderland"
```

For more complex rendering (conditionals, loops, includes), plug in a custom renderer:

```python
from lllm.core.models import BaseRenderer

class JinjaRenderer(BaseRenderer):
    def __init__(self, env=None):
        from jinja2 import Environment
        self.env = env or Environment()

    def render(self, prompt, **kwargs):
        template = self.env.from_string(prompt)
        return template.render(**kwargs)

prompt = Prompt(
    path="complex",
    prompt="{% for item in items %}{{ item }}\n{% endfor %}",
    renderer=JinjaRenderer(),
)
```

The framework takes no dependency on Jinja or any other template engine — the `BaseRenderer` interface is the extension point.

## Prompt Composition

The `extend` method creates a child prompt that inherits all fields from the parent, with specified overrides:

```python
base = Prompt(
    path="base/analyst",
    prompt="You are a research analyst.\n\n{instruction}",
    parser=DefaultTagParser(xml_tags=["answer"], required_xml_tags=["answer"]),
    function_list=[search_tool, calculator_tool],
)

# Child inherits parser, tools, handlers — only overrides template and path
focused = base.extend(
    path="finance/analyst",
    prompt="You are a financial analyst.\n\n{instruction}",
)

# Child with different tools
lightweight = base.extend(
    path="base/analyst_no_tools",
    function_list=[],
)
```

A new `path` is always required to prevent registry collisions.

## Handlers

Handlers define how the agent loop recovers from errors and processes tool results. Each handler can be a simple string template or a full Prompt for advanced customization.

**Default** — string templates auto-wrap into lightweight Prompts:

```python
Prompt(
    path="bot",
    prompt="...",
    on_exception="Error: {error_message}. Please fix your response.",
    on_interrupt="Tool returned: {call_results}. Continue.",
    on_interrupt_final="All tools done. Give your final answer.",
)
```

**Custom** — provide a full Prompt with its own parser, tools, or behaviour:

```python
error_recovery = Prompt(
    path="bot/error_recovery",
    prompt="Your response had this error: {error_message}. "
           "Use the validator tool to verify your fix.",
    parser=DefaultTagParser(xml_tags=["fix"], required_xml_tags=["fix"]),
    function_list=[validator_tool],
)

Prompt(
    path="bot",
    prompt="...",
    on_exception=error_recovery,
)
```

Handlers are resolved once (via `@cached_property`) and reused across the agent loop. String handlers inherit the parent's parser and output contract; custom Prompt handlers use their own.

## Provider Capabilities

Provider-specific features live in the `addon_args` dict rather than as dedicated fields. This ensures new provider features never require schema changes:

```python
Prompt(
    path="browser_agent",
    prompt="...",
    addon_args={
        "web_search": True,
        "computer_use": {"display_width": 1280, "display_height": 800},
    },
)
```

Convenience properties provide readable access:

```python
prompt.allow_web_search      # bool, reads addon_args["web_search"]
prompt.computer_use_config   # dict, reads addon_args["computer_use"]
```

The invoker reads `addon_args` and translates entries to provider-specific tool configurations. Prompts stay provider-agnostic.

## Metadata and Tracking

Every prompt can carry arbitrary metadata via the `meta` field, and can produce a serializable snapshot for experiment tracking:

```python
prompt = Prompt(
    path="analyst",
    prompt="...",
    meta={"author": "junyan", "experiment": "ablation-no-cot"},
)

prompt.to_metadata()
# {
#     "path": "analyst",
#     "prompt_hash": "a1b2c3d4e5f6",
#     "meta": {"author": "junyan", "experiment": "ablation-no-cot"},
#     "functions": ["search", "calculator"],
#     "mcp_servers": [],
#     "addon_args": ["web_search"],
#     "has_parser": True,
#     "has_format": False,
# }
```

This metadata flows through to the logging system and can be consumed by external tracking tools (wandb, mlflow, etc.) without any framework integration.