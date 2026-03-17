# Lesson 4 — Tools: Giving Agents Superpowers

Tools let an agent call Python functions during a conversation. The LLM decides when to invoke a tool, receives the result, and continues reasoning. LLLM handles the interrupt loop automatically.

---

## The `@tool` Decorator

The fastest way to define a tool is with the `@tool` decorator:

```python
from lllm.core.prompt import tool

@tool(
    description="Get the current weather for a city",
    prop_desc={
        "city": "The city name, e.g. 'Paris'",
        "unit": "Temperature unit: 'celsius' or 'fahrenheit'",
    },
)
def get_weather(city: str, unit: str = "celsius") -> str:
    # Replace with a real API call
    return f"Sunny, 22°C in {city}"
```

`@tool` inspects the function signature to build the JSON Schema that gets sent to the model. Type hints map to JSON types:

| Python | JSON |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |

Parameters without defaults are marked as `required` automatically.

---

## Attaching Tools to a Prompt

```python
from lllm import Prompt

weather_prompt = Prompt(
    path="weather_bot/system",
    prompt="You are a weather assistant. Use your tools to answer questions.",
    function_list=[get_weather],   # list of Function objects
)
```

When the agent uses this prompt, the model sees the tool schema and can call `get_weather`.

---

## Full Example: Weather Bot

```python
from lllm import Tactic
from lllm.core.prompt import tool, Prompt
from lllm.invokers import build_invoker
from lllm import Agent

@tool(description="Get current weather for a city")
def get_weather(city: str, unit: str = "celsius") -> str:
    return f"Sunny, 22°C in {city}"

weather_prompt = Prompt(
    path="weather_bot/system",
    prompt="You are a weather assistant.",
    function_list=[get_weather],
)

invoker = build_invoker({"invoker": "litellm"})
agent = Agent(
    name="weather_bot",
    system_prompt=weather_prompt,
    model="gpt-4o",
    llm_invoker=invoker,
    max_interrupt_steps=3,   # allow up to 3 tool calls per response
)

agent.open("chat")
agent.receive("What is the weather in Paris and Tokyo?")
response = agent.respond()
print(response.content)
```

The agent loop:
1. LLM sees the user question and decides to call `get_weather(city="Paris")`.
2. LLLM executes the function and feeds the result back.
3. LLM calls `get_weather(city="Tokyo")`.
4. LLLM executes and feeds back.
5. LLM writes the final answer.
6. `respond()` returns it.

---

## Controlling the Tool-Call Loop

```python
agent = Agent(
    ...
    max_interrupt_steps=5,   # maximum tool calls before forcing a final answer
    max_exception_retry=3,   # maximum retries when the model output fails validation
)
```

- `max_interrupt_steps=0` means unlimited (up to 100). Avoid this in production — always set an explicit cap.
- When the limit is reached, the agent appends a message asking the model to stop calling tools and write a final answer.

---

## Custom Result Formatting

By default the tool result is formatted as:

```
Return of calling function get_weather with arguments {'city': 'Paris', 'unit': 'celsius'}:
---
Sunny, 22°C in Paris
---
```

You can override this with a custom `processor`:

```python
def my_processor(result, function_call):
    return f"[{function_call.name}] → {result}"

@tool(description="Get weather", processor=my_processor)
def get_weather(city: str) -> str:
    return "Sunny"
```

---

## Building a `Function` Manually

For tools with complex schemas or when you want to declare the schema separately from the implementation:

```python
from lllm import Prompt
from lllm.core.prompt import Function

search_fn = Function(
    name="web_search",
    description="Search the web for information",
    properties={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Max results (default 5)"},
    },
    required=["query"],
)

# Attach the implementation separately
search_fn.link_function(lambda query, max_results=5: f"Top {max_results} results for '{query}'")

my_prompt = Prompt(
    path="research_bot/system",
    prompt="You are a research assistant.",
    function_list=[search_fn],
)
```

---

## MCP Servers

For tools exposed via the Model Context Protocol (MCP):

```python
from lllm.core.prompt import MCP, Prompt

mcp_server = MCP(
    server_label="my_tools",
    server_url="http://localhost:8080",
    require_approval="never",
    allowed_tools=["search", "fetch_url"],
)

my_prompt = Prompt(
    path="agent/system",
    prompt="You are an assistant with web access.",
    mcp_servers_list=[mcp_server],
)
```

> **Note:** MCP support requires an MCP-compatible invoker. Check the [Invokers reference](../reference/invokers.md) for details.

---

## Repeated Tool Call Detection

If the agent calls the same tool with the same arguments twice in one turn, LLLM detects the repetition and injects a warning telling the model not to call it again. This prevents infinite loops caused by a confused model.

---

## Error Handling in Tools

If your tool function raises an exception, LLLM catches it and feeds the error message back to the model as the tool result:

```python
@tool(description="Divide two numbers")
def divide(a: float, b: float) -> str:
    if b == 0:
        raise ValueError("Division by zero")
    return str(a / b)
```

The model sees `"Error: Division by zero"` as the result and can decide how to proceed (e.g., ask for different arguments).

---

---

## Proxy Tools — Structured Access to External APIs

For agents that need to call many API endpoints, the `@tool` approach scales poorly: dozens of endpoints means dozens of tool definitions and a bloated context. LLLM's **proxy system** solves this with a two-tool pattern the agent uses lazily:

- **`query_api_doc(proxy_name)`** — retrieve full endpoint documentation on demand
- **`run_python(code)`** — execute Python in a persistent interpreter with `CALL_API` pre-injected

The agent learns the endpoint structure just-in-time, then calls whatever it needs through `CALL_API` in Python. LLLM injects both tools and the supporting prompt block automatically when you declare a proxy config — no extra wiring needed.

### Step 1: Define a proxy

A proxy wraps an API surface. Endpoints are declared with `@BaseProxy.endpoint` — this metadata drives the API directory, `query_api_doc` responses, and auto-testing.

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(
    path="weather",
    name="Weather API",
    description="Current weather and forecasts for any city.",
)
class WeatherProxy(BaseProxy):

    @BaseProxy.endpoint(
        category="current",
        endpoint="conditions",
        description="Current temperature and conditions for a city.",
        params={
            "city*": (str, "London"),       # * = required
            "units": (str, "celsius"),
        },
        response={"temperature": 18, "condition": "cloudy", "humidity": 72},
    )
    def conditions(self, params: dict) -> dict:
        city = params["city"]
        return {"temperature": 18, "condition": "cloudy", "humidity": 72}

    @BaseProxy.endpoint(
        category="forecast",
        endpoint="daily",
        description="5-day daily forecast for a city.",
        params={
            "city*": (str, "London"),
            "days": (int, 5),
        },
        response=[{"date": "2024-01-01", "high": 20, "low": 12}],
    )
    def daily(self, params: dict) -> list:
        city = params["city"]
        return [{"date": f"2024-01-0{i}", "high": 18+i, "low": 10+i} for i in range(1, 6)]
```

Key conventions:
- `path` in `@ProxyRegistrator` is what you put in `activate_proxies`
- Parameters with `*` suffix are required; the second tuple element is the example value
- `response` is the example response shown in `query_api_doc` output

### Step 2: Configure the agent

Add a `proxy:` block to the agent config. With `exec_env: interpreter` (the default), LLLM builds a `ProxyManager`, an `AgentInterpreter`, and injects both tools automatically:

```yaml
# config.yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: weather_analyst
    system_prompt: "You are a weather analyst. Use the Weather API to answer questions."
    proxy:
      activate_proxies: [weather]
      exec_env: interpreter       # default — can be omitted
      max_output_chars: 5000
      timeout: 60.0
```

Or in Python:

```python
config = {
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{
        "name": "weather_analyst",
        "system_prompt": "You are a weather analyst.",
        "proxy": {
            "activate_proxies": ["weather"],
            "exec_env": "interpreter",
            "max_output_chars": 5000,
        },
    }],
}
```

### Step 3: Run it

```python
from lllm import Tactic

class WeatherTactic(Tactic):
    name = "weather"
    agent_group = ["weather_analyst"]

    def call(self, task: str) -> str:
        agent = self.agents["weather_analyst"]
        agent.open("main")
        agent.receive(task)
        return agent.respond().content

tactic = WeatherTactic(config)
print(tactic("What is the current weather in Tokyo and Paris?"))
```

The agent will typically:
1. Call `query_api_doc("weather")` to see full endpoint specs
2. Call `run_python` with code that calls `CALL_API("weather/conditions", {"city": "Tokyo"})`
3. Call `run_python` again (reusing variables) to process both cities and format the answer

### How the interpreter works

`AgentInterpreter` uses a persistent namespace passed to every `exec()` call, so variables survive across calls like a live REPL:

```python
# Agent turn 1: fetch data
data_tokyo = CALL_API("weather/conditions", {"city": "Tokyo"})
data_paris = CALL_API("weather/conditions", {"city": "Paris"})
print(data_tokyo)
```

```python
# Agent turn 2: data_tokyo is still in scope
comparison = f"Tokyo: {data_tokyo['temperature']}°C, Paris: {data_paris['temperature']}°C"
print(comparison)
```

- **Stdout capture** — only `print()` output is captured; the last expression is not auto-returned.
- **Exception handling** — uncaught exceptions are returned as formatted tracebacks so the agent can diagnose and retry without losing session state.
- **Truncation** — output longer than `max_output_chars` is cut off with a truncation indicator. The agent is told this in the injected prompt block.
- **Timeout** — each `run_python` call runs in a daemon thread; `TimeoutError` is raised if it exceeds `timeout` seconds.

### Global vs per-agent proxy config

```yaml
global:
  model_name: gpt-4o
  proxy:
    activate_proxies: [fmp, fred]
    exec_env: interpreter

agent_configs:
  - name: fast_analyst
    # inherits global proxy config

  - name: notebook_analyst
    proxy:
      exec_env: jupyter      # override: use JupyterSession instead

  - name: crypto_analyst
    proxy:
      activate_proxies: [fmp]   # override: only fmp
      timeout: 30.0
```

Per-agent values deep-merge on top of global, field by field.

### Full proxy config reference

```yaml
proxy:
  activate_proxies: [fmp, fred]   # which proxies to load; empty = all registered
  deploy_mode: false               # passed to proxy instances (e.g. for rate limiting)
  cutoff_date: "2024-01-01"        # ISO date; proxies can use this to filter data
  exec_env: interpreter            # "interpreter" | "jupyter" | null
  max_output_chars: 5000           # truncate run_python stdout; 0 = no limit
  truncation_indicator: "... (truncated)"
  timeout: 60.0                    # seconds per run_python call; interpreter only
  prompt_template: null            # custom template string; null = auto-select
```

### Interpreter vs Jupyter mode

Use `exec_env: jupyter` when you need notebook files as output artifacts, Matplotlib/Plotly visualizations, or cell-level execution history. In Jupyter mode, only `query_api_doc` is injected as a tool — the agent writes `<python_cell>` XML tags that your tactic runs in a `JupyterSession`.

| | Interpreter | Jupyter |
|--|-------------|---------|
| Overhead | None (in-process exec) | Subprocess per kernel (~2–3s startup) |
| Parallel agents | Safe — isolated namespaces | Heavy — one subprocess per agent |
| Visualizations | Not available | Full matplotlib/plotly support |
| Output artifact | None | `.ipynb` notebook file |
| Audit trail | None | Cell-by-cell execution history |
| Best for | Data retrieval, computation, API orchestration | Exploratory analysis, charts, reports |

See [Proxies & Sandbox reference](../core/proxy-and-sandbox.md) for a complete Jupyter mode example.

---

## Jupyter Notebook Sandbox — Step by Step

`JupyterSession` is LLLM's **built-in sandbox**, optimised for the proxy system. Code executes in a real kernel, plots are saved, variables persist across turns, and the session produces a `.ipynb` file as an output artifact.

Unlike interpreter mode (where LLLM runs code in-process), in Jupyter mode the split of responsibility is explicit: the **agent** writes `<python_cell>` / `<markdown_cell>` XML tags; your **tactic** extracts those cells and runs them in a `JupyterSession`; execution results are fed back so the agent can continue. This lets you intercept every cell, handle errors, and checkpoint state.

### Step 1: Write the system prompt and parser

The system prompt explains the notebook interface. A parser enforces the tag format:

```python
from lllm import Prompt
from lllm.utils import find_all_xml_tags_sorted
from lllm.core.const import ParseError

NOTEBOOK_SYSTEM = """
You are a data analyst working in a Jupyter Notebook.

## Interface
- Write Python in `<python_cell>...</python_cell>` tags.
- Write notes and summaries in `<markdown_cell>...</markdown_cell>` tags.
- When your analysis is complete, respond with `<TERMINATE_NOTEBOOK>` alone (no cells in that response).

Rules:
- Cells execute sequentially; variables persist across turns.
- If a cell fails you will see the traceback and must fix that cell only.
- Use `CALL_API(path, params)` to call data APIs — always call `query_api_doc` first.
"""

def notebook_parser(message: str, **kwargs):
    matches = find_all_xml_tags_sorted(message)
    terminate = "<TERMINATE_NOTEBOOK>" in message
    cells = [
        (m["tag"], m["content"])
        for m in matches
        if m["tag"] in ("python_cell", "markdown_cell")
    ]
    if not cells and not terminate:
        raise ParseError(
            "No cells found. Wrap code in <python_cell>...</python_cell> "
            "or markdown in <markdown_cell>...</markdown_cell>, "
            "or respond with <TERMINATE_NOTEBOOK>."
        )
    # If TERMINATE_NOTEBOOK is mixed with python cells, ignore the termination signal
    if terminate and any(tag == "python_cell" for tag, _ in cells):
        terminate = False
    return {"raw": message, "cells": cells, "terminate": terminate}

analyst_system = Prompt(
    path="analyst/system",
    prompt=NOTEBOOK_SYSTEM,
    xml_tags=["python_cell", "markdown_cell"],
    signal_tags=["TERMINATE_NOTEBOOK"],
    parser=notebook_parser,
)
```

### Step 2: Configure the agent

Set `exec_env: jupyter` so LLLM injects only `query_api_doc` (not `run_python` — the tactic drives execution):

```yaml
# config.yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: analyst
    system_prompt_path: analyst/system
    proxy:
      activate_proxies: [weather]
      exec_env: jupyter
```

### Step 3: Write the tactic

The tactic drives the loop: send task → get cells → run cells → feed output back → repeat:

```python
from lllm import Tactic
from lllm.sandbox.jupyter import JupyterSession

class NotebookAnalysisTactic(Tactic):
    name = "notebook_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs) -> str:
        agent = self.agents["analyst"]

        # Initialise a JupyterSession — CALL_API is wired into the kernel automatically
        session = JupyterSession(
            name="analysis",
            dir="/tmp/notebooks",
            metadata={
                "proxy": {
                    "activate_proxies": ["weather"],
                    "deploy_mode": False,
                }
            },
        )
        session.init_session()   # injects CALL_API into the kernel namespace

        agent.open("main")
        agent.receive(task)

        for _ in range(10):          # max steps
            msg = agent.respond()
            parsed = msg.parsed

            if parsed["terminate"]:
                break

            # Execute each cell and collect output
            outputs = []
            for tag, code in parsed["cells"]:
                if tag == "python_cell":
                    output = session.run_cell(code)
                    outputs.append(f"[output]\n{output}")
                else:
                    session.add_markdown_cell(code)

            # Feed results back so the agent can continue
            if outputs:
                agent.receive("\n\n".join(outputs))

        return session.notebook_path   # path to the saved .ipynb file
```

The agent typically:
1. Calls `query_api_doc("weather")` via the injected tool to see endpoint specs.
2. Writes a `<python_cell>` that calls `CALL_API(...)` — the tactic runs it and returns stdout.
3. Receives the output, writes more cells to analyse and visualise the data.
4. Responds with `<TERMINATE_NOTEBOOK>` when done.

### Step 4: Handle cell errors

If a cell fails, feed the traceback back so the agent can fix just that cell:

```python
for tag, code in parsed["cells"]:
    if tag == "python_cell":
        output = session.run_cell(code)
        if session.last_cell_failed:
            # Agent sees the error and provides a replacement cell
            agent.receive(
                f"Cell failed with the following error:\n\n{output}\n\n"
                "Please fix the error. Provide one <python_cell> with the corrected code only."
            )
            fix_msg = agent.respond()
            _, fixed_code = fix_msg.parsed["cells"][0]
            output = session.run_cell(fixed_code)
        outputs.append(f"[output]\n{output}")
```

### Bringing your own execution environment

`JupyterSession` is the built-in sandbox, but the pattern — agent writes tags, tactic runs code — works with any executor. The prompt and parser stay unchanged; only the sandbox object changes:

```python
class DockerSandbox:
    """Minimal example: run cells in an isolated container."""

    def run_cell(self, code: str) -> str:
        import subprocess
        result = subprocess.run(
            ["docker", "exec", "my_sandbox", "python3", "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout or result.stderr

    def add_markdown_cell(self, text: str) -> None:
        pass   # no-op for non-notebook environments

    last_cell_failed: bool = False   # set after run_cell if needed


class DockerAnalysisTactic(Tactic):
    name = "docker_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs) -> str:
        agent = self.agents["analyst"]
        sandbox = DockerSandbox()

        agent.open("main")
        agent.receive(task)

        for _ in range(10):
            msg = agent.respond()
            parsed = msg.parsed
            if parsed["terminate"]:
                break
            outputs = []
            for tag, code in parsed["cells"]:
                if tag == "python_cell":
                    outputs.append(sandbox.run_cell(code))
                else:
                    sandbox.add_markdown_cell(code)
            if outputs:
                agent.receive("\n\n".join(outputs))

        return agent.current_dialog.tail.content
```

You could replace `DockerSandbox` with a remote code execution service, a WebAssembly sandbox, a database query runner — anything that accepts code strings and returns output strings.

See [Proxies & Sandbox reference](../core/proxy-and-sandbox.md) for the full `JupyterSession` API, `init_session` wiring details, and a complete Jupyter mode example.

---

## Summary

| Concept | API |
|---|---|
| Define a tool | `@tool(description=..., prop_desc={...})` |
| Attach to prompt | `Prompt(function_list=[my_fn])` |
| Control loop depth | `Agent(max_interrupt_steps=N)` |
| Custom result format | `@tool(processor=my_fn)` |
| Manual schema | `Function(name=..., properties={...})` |
| MCP server | `Prompt(mcp_servers_list=[MCP(...)])` |
| Define a proxy | `@ProxyRegistrator` + `@BaseProxy.endpoint` |
| Wire proxy to agent | `proxy: {activate_proxies: [...], exec_env: interpreter}` |
| Lazy API docs | `query_api_doc(proxy_name)` (auto-injected) |
| Execute code | `run_python(code)` with `CALL_API` (auto-injected) |
| Jupyter sandbox | `exec_env: jupyter` + `JupyterSession` in tactic |
| Custom sandbox | Replace `JupyterSession` with any object that has `run_cell()` |

**Next:** [Lesson 5 — Tactics: Coordinating Multiple Agents](05-tactics.md)
