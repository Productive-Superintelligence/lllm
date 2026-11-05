# 4. Tools, Proxies, and Sandbox

**Goal:** give an agent Python tools and watch the tool-call loop, control that
loop, declare tools separately from their implementations, expose a whole
tactic or an MCP server as a tool, connect research data APIs through proxies,
and run agent-written code in a sandbox you control.

LLLM gives you three levels of capability, each with more autonomy:

- **Tools**: Python functions the model can call during a turn. LLLM runs the
  tool-call loop for you, and you decide which tools are on offer, how results
  are reported, and how many rounds are allowed.
- **Proxies**: whole data APIs (dozens of endpoints) behind a small programming
  surface. Instead of one schema per endpoint, the agent reads the API's
  documentation and calls endpoints from Python with `CALL_API`; the proxy adds
  credentials and sends the requests. This is how LLLM agents work with
  research data sources such as financial statements or economic series.
- **Sandboxes**: the agent writes code cells and your tactic decides where and
  how they run.

Where Pydantic fits: tool schemas are generated from type annotations,
including lists, `Literal` choices, and nested Pydantic models. Tool schemas
(`Function`) and tool invocations (`FunctionCall`) are Pydantic models
themselves, and when a tactic is used as a tool, its arguments are validated
against the tactic's `input_type` ([Lesson 5](05-native-tactics.md)).

## Prerequisites

Finish [Lesson 3](03-prompts.md), in particular Step 3: the most recent prompt
put into a dialog drives the next `respond()`. Steps that call `respond()` or
run a tactic need a model provider key; defining tools, proxies, and sandboxes
and calling them directly runs locally.

---

## Part 1: Tools

### Step 1: Define a tool with `@tool`

```python
from lllm import tool


@tool(
    description="Get the current weather for a city",
    prop_desc={
        "city": "The city name, e.g. 'Paris'",
        "unit": "Temperature unit: 'celsius' or 'fahrenheit'",
    },
)
def get_weather(city: str, unit: str = "celsius") -> str:
    # Replace with a real weather API call.
    return f"Sunny, 22 degrees {unit} in {city}"
```

`@tool` turns the function into a `Function`: the schema the model sees plus the
implementation. It builds the JSON Schema from the signature:

```python
import json

print(json.dumps(get_weather.to_tool(), indent=2))
```

```text
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "The city name, e.g. 'Paris'"
        },
        "unit": {
          "type": "string",
          "description": "Temperature unit: 'celsius' or 'fahrenheit' (default: 'celsius')"
        }
      },
      "required": [
        "city"
      ],
      "additionalProperties": false
    },
    "strict": false
  }
}
```

- Parameters without defaults are required; defaults are noted in the
  description.
- Without `description`, the function's docstring is used.
- `strict` is chosen for you: a schema is marked strict only when every
  parameter is required and flat, which is what strict function calling (for
  example OpenAI's) accepts. Pass `strict=True` or `strict=False` to decide
  yourself.

Annotations map to JSON Schema, including containers, choices, and models;
unannotated parameters become strings:

```python
from typing import Literal

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


@tool(description="Draw a line through the points")
def draw(points: list[Point], color: Literal["red", "blue"]) -> str:
    return f"Drew {len(points)} points in {color}"


print(json.dumps(draw.properties["color"]))   # {"enum": ["red", "blue"], "type": "string"}
print(draw.properties["points"]["type"], draw.properties["points"]["items"]["required"])  # array ['x', 'y']
print(draw.strict)                            # False: the points are nested objects
```

| Annotation | JSON Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[str]` | an `array` of strings |
| `dict[str, int]` | an `object` with integer values |
| `Literal["red", "blue"]` | a string `enum` |
| `Optional[int]` | `anyOf` integer or null |
| a Pydantic model | the model's object schema, inlined |

### Step 2: Call a tool yourself

A tool is called with a `FunctionCall`, exactly as the agent loop does:

```python
from lllm import FunctionCall

call = get_weather(FunctionCall(name="get_weather", arguments={"city": "Paris"}))
print(call.result)       # Sunny, 22 degrees celsius in Paris
print(call.result_str)
```

`result_str` is what the model reads back:

```text
Return of calling function get_weather with arguments {'city': 'Paris'}:
---
Sunny, 22 degrees celsius in Paris
---
```

### Step 3: Decide where tools live

Tools are declared on prompts, in one of two places:

- **The agent's system prompt.** These are the agent's own tools, offered on
  every turn, whether the turn was sent with `receive(text)` or with a prompt.
- **The prompt for one turn**, sent with `receive_prompt()`. Its tools are
  offered on that turn in addition to the agent's own; if both declare a tool
  with the same name, the turn's version wins. Set `agent_tools=False` on a turn
  prompt to offer only that prompt's tools (or none).

```python
from lllm import Prompt

weather_system = Prompt(
    path="weather_bot/system",
    prompt="You are a weather assistant.",
    function_list=[get_weather],
)
print(sorted(weather_system.functions))   # ['get_weather']
```

### Step 4: Run the tool loop

```python
from lllm import Agent
from lllm.runtimes.native.invokers import build_invoker

agent = Agent(
    name="weather_bot",
    system_prompt=weather_system,
    model="gpt-4o",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    max_interrupt_steps=3,   # at most 3 rounds of tool calls per respond()
)

agent.open("chat")
agent.receive("What is the weather in Paris and Tokyo?")
response = agent.respond()
print(response.content)

for message in agent.current_dialog.messages:
    print(f"{message.role.value:>9} | {message.name}")
```

A typical run leaves this in the dialog (the model may request both lookups in
one round, as here, or one per round):

```text
   system | system
     user | user
tool_call | weather_bot
     tool | get_weather
     tool | get_weather
assistant | weather_bot
```

What `respond()` did:

1. The model saw the question and the `get_weather` schema, and replied with
   tool calls instead of text.
2. LLLM ran each requested function and appended the results as `tool`
   messages.
3. The model was called again with the results and wrote the final answer.
4. `respond()` returned that final `Message`.

A turn prompt can add a tool for one turn, or switch the agent's tools off:

```python
@tool(description="Convert a temperature from Celsius to Fahrenheit")
def to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


convert = Prompt(path="weather_bot/convert", prompt="{question}", function_list=[to_fahrenheit])
agent.receive_prompt(convert, {"question": "How warm is Paris today in Fahrenheit?"})
print(agent.respond().content)    # get_weather and to_fahrenheit are both offered

recap = Prompt(path="weather_bot/recap", prompt="Summarize our conversation in one sentence.", agent_tools=False)
agent.receive_prompt(recap)
print(agent.respond().content)    # no tools are offered on this turn
```

### Step 5: Control the loop

```python
@tool(description="Divide two numbers")
def divide(a: float, b: float) -> str:
    if b == 0:
        raise ValueError("Division by zero")
    return str(a / b)


print(divide(FunctionCall(name="divide", arguments={"a": 1, "b": 0})).result_str)
# Error: Division by zero
```

The loop is governed by a few rules:

- **Round limit.** `max_interrupt_steps` (default 5) caps tool-call rounds per
  `respond()`. When the cap is reached, LLLM appends "You are reaching the limit
  of tool calls. Provide the final response." and calls the model once more
  without tools. `0` means up to 100 rounds and logs a warning; always set a cap.
- **Tool errors.** An exception inside a tool becomes the result
  `Error: <message>`, as above, so the model can retry with other arguments.
- **Repeated calls.** If the model repeats a call with identical arguments in
  the same `respond()`, LLLM does not run it again and tells the model to use
  the earlier result.
- **Unknown tools.** A call to a tool that is not offered on the turn raises
  `KeyError`.

To see the rounds of one call, ask for its session:

```python
agent.receive("And in Berlin?")
session = agent.respond(return_session=True)

for step, rounds in session.interrupts.items():
    for calls in rounds:
        print(step, [(call.name, call.arguments) for call in calls])
```

### Step 6: Format tool results

The default result text is the "Return of calling function ..." block from
Step 2. Pass a `processor(result, function_call) -> str` to change it:

```python
def short_result(result, function_call):
    return f"[{function_call.name}] {result}"


@tool(description="Get the current weather for a city", processor=short_result)
def weather_brief(city: str) -> str:
    return f"Sunny in {city}"


print(weather_brief(FunctionCall(name="weather_brief", arguments={"city": "Paris"})).result_str)
# [weather_brief] Sunny in Paris
```

### Step 7: Declare a tool separately from its implementation

Sometimes the prompt should own the interface while the implementation lives
elsewhere, like a header file and its source. Declare the schema as a
`Function` without an implementation:

```python
from lllm import Function

search_decl = Function(
    name="web_search",
    description="Search the web for information",
    properties={"query": {"type": "string", "description": "Search query"}},
    required=["query"],
)
```

Then register an implementation under the same name in a runtime registry:

```python
from lllm import Runtime


@tool(name="web_search", description="Search the web for information")
def web_search(query: str) -> str:
    return f"Top results for {query!r}"   # call your search API here


runtime = Runtime()
runtime.register_tool("web_search", web_search)
```

An agent resolves declarations against its runtime when a turn starts:

```python
researcher = Agent(
    name="researcher",
    system_prompt=Prompt(
        path="research_bot/system",
        prompt="You research questions carefully.",
        function_list=[search_decl],
    ),
    model="gpt-4o",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    runtime=runtime,
)

researcher.open("main")
researcher.receive("What is the LLLM framework?")
print(researcher.respond().content)
```

Binding is by exact name. If nothing named `web_search` is registered,
`respond()` raises `KeyError` naming the missing tool. A plain string works as
well, `function_list=["web_search"]`, and a prompt can mix live `Function`
objects, declarations, and strings in one list. Inside an LLLM package, tools
defined in the package's tool folders are registered automatically and can be
referenced by package refs such as `"my_pkg.tools:web_search"`
([Lesson 6](06-packages-and-hub.md) explains how refs are formed).

### Step 8: Use a tactic as a tool

A tool can be a whole agentic workflow. `tactic_as_function()` turns a tactic
into a `Function` whose single `task` argument is passed to the tactic:

```python
from lllm import Prompt, Tactic
from lllm.runtimes.native import tactic_as_function


class CodeReviewTactic(Tactic):
    """Review a Python snippet and list concrete problems."""

    name = "code_review"
    agent_group = ["reviewer"]
    input_type = str
    output_type = str

    def call(self, task: str) -> str:
        reviewer = self.agents["reviewer"]
        reviewer.open("review")
        reviewer.receive(f"Review this code and list concrete problems:\n\n{task}")
        return reviewer.respond().content


review_tool = tactic_as_function(CodeReviewTactic({
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "reviewer", "system_prompt": "You are a strict Python reviewer."}],
}))

print(review_tool.name, review_tool.required)   # code_review ['task']
print(review_tool.description)                  # the tactic's docstring

ask_dev = Prompt(path="dev/ask", prompt="{request}", function_list=[review_tool])
```

Tactics are covered in [Lesson 5](05-native-tactics.md). Tactics can also mark
individual methods as tools with `@tactictool("name", ...)` and be referenced by
package ref, for example `"my_pkg.tactics:code_review"`.

### Step 9: Connect MCP servers

Tools served over the Model Context Protocol are declared with `MCP` and
attached with `mcp_servers_list`, on the system prompt (every turn) or on a
turn prompt:

```python
from lllm import MCP, Prompt

docs_server = MCP(
    server_label="docs",
    server_url="https://example.com/mcp",
    require_approval="never",
    allowed_tools=["search", "fetch_url"],
)

ask_docs = Prompt(path="docs_bot/ask", prompt="{question}", mcp_servers_list=[docs_server])
print(docs_server.to_tool())
```

The provider calls MCP servers itself; LLLM passes the declaration along as a
`{"type": "mcp", ...}` tool. Whether that is accepted depends on the provider
and API. For example, OpenAI exposes remote MCP servers through its Responses
API, which you select with `api_type=APITypes.RESPONSE` on the agent (import
`APITypes` from `lllm.runtimes.native`).

### Step 10: Declare agent tools in configuration

When agents are built from configuration ([Lesson 5](05-native-tactics.md)),
a `tools` key lists capabilities for an agent, or for all agents under
`global`. The refs are added to the agent's system prompt, resolved from the
tactic's registry, and offered on every turn like any other agent tool:

```python
from lllm import Runtime, Tactic

runtime = Runtime()
runtime.register_tool("web_search", web_search)


class AnalystTactic(Tactic):
    name = "analyst"
    agent_group = ["analyst"]

    def call(self, task: str) -> str:
        analyst = self.agents["analyst"]
        analyst.open("main")
        analyst.receive(task)
        return analyst.respond().content


config = {
    "global": {"model_name": "gpt-4o", "tools": ["web_search"]},
    "agent_configs": [{"name": "analyst", "system_prompt": "You are a research analyst."}],
}

analyst = AnalystTactic(config, registry=runtime)
print(analyst("Find recent work on program synthesis with LLMs."))
```

Supported `tools` entries are registered tool names or `pkg.tools:name` refs,
tactic tools (`pkg.tactics:name`, or `pkg.tactics:name#tool` to pick one
method), and proxies (`pkg.proxies:name`), which add the proxy tools described
next.

---

## Part 2: Proxies

When an agent needs many endpoints, one function schema per endpoint bloats
the tool list and still cannot express multi-step data work. A **proxy** wraps
an API surface instead: the agent reads the API's documentation and works with
it from Python, through two tools:

- `query_api_doc(proxy_name)` returns a proxy's endpoint documentation.
- `run_python(code)` runs Python in a persistent interpreter where
  `CALL_API(path, params)` is already available.

The agent then fetches, joins, and computes over the data in ordinary Python,
in a few tool rounds.

### Step 11: Define a proxy

```python
from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="weather",
    name="Weather API",
    description="Current weather and forecasts for any city.",
)
class WeatherProxy(BaseProxy):

    @BaseProxy.endpoint(
        category="current",
        endpoint="current/{city}",
        description="Current temperature and conditions for a city.",
        params={
            "city*": (str, "London"),     # * marks a required parameter
            "units": (str, "celsius"),
        },
        response={"temperature": 18, "condition": "cloudy"},
    )
    def current(self, params: dict) -> dict:
        return {"city": params["city"], "temperature": 18, "condition": "cloudy"}

    @BaseProxy.endpoint(
        category="forecast",
        endpoint="forecast/daily",
        description="Daily forecast for a city.",
        params={
            "city*": (str, "London"),
            "days": (int, 5),
        },
        response=[{"day": 1, "high": 20, "low": 12}],
    )
    def daily_forecast(self, params: dict) -> list:
        days = params.get("days", 5)
        return [{"day": i + 1, "high": 18 + i, "low": 10 + i} for i in range(days)]
```

- `@ProxyRegistrator` registers the class in the default runtime under `path`,
  the name you activate and the first segment of every `CALL_API` path.
- `endpoint` is the path the agent calls, exactly as the documentation lists
  it: `CALL_API("weather/forecast/daily", {"city": "Tokyo"})`. Paths can
  contain slashes, hyphens, and `{name}` placeholders, which are filled from
  the path: `CALL_API("weather/current/Tokyo", {})` passes `city="Tokyo"`.
- Only methods decorated with `@BaseProxy.endpoint` can be called.
- This proxy has no `base_url`, so each endpoint method computes the response
  itself. Step 12 proxies a real HTTP API.
- In `params`, the tuple is `(type, example value)`. `response` is an example
  response. Both appear in the documentation the agent reads.

### Step 12: Proxy an HTTP API

For a web API, declare where it lives and how to authenticate, and let each
endpoint method prepare the request parameters:

```python
import os

from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="weather_api",
    name="Weather HTTP API",
    description="Current weather from a JSON web API.",
)
class WeatherHTTPProxy(BaseProxy):
    base_url = "https://api.example.com/weather"
    api_key_name = "api_key"   # sent as a query parameter; "*X-Api-Key" would send a header

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("WEATHER_API_KEY")

    @BaseProxy.endpoint(
        category="current",
        endpoint="current/{city}",
        description="Current weather for a city.",
        params={"city*": (str, "London"), "units": (str, "metric")},
        response={"temperature": 18.2, "condition": "cloudy"},
    )
    def current(self, params: dict) -> dict:
        params.setdefault("units", "metric")   # validate or adjust the request here
        return params
```

With a `base_url`, an endpoint method returns the parameters to send, and the
proxy makes the request: `CALL_API("weather_api/current/Tokyo", {})` sends
`GET https://api.example.com/weather/current/Tokyo?units=metric&api_key=...`.

- `{name}` placeholders in the endpoint path are filled in the URL.
- The key is added under `api_key_name`; a name starting with `*` sends it as a
  header instead.
- Endpoints declared with `method="POST"` send a JSON body.
- Responses are cached on disk under `~/.lllm` by default; pass
  `use_cache=False` to `super().__init__()` to turn that off.
- Override `_call_api(url, params, endpoint_info, headers)` for APIs that need
  extra request fields or response post-processing.

The proxies that ship with LLLM (Step 15) are written this way.

### Step 13: Call proxies from Python

`ProxyManager` loads activated proxies and dispatches calls; it is exactly what
the agent's `CALL_API` uses:

```python
from lllm import ProxyManager
from lllm.runtimes.native.proxies import AgentInterpreter

proxies = ProxyManager(activate_proxies=["weather"])
print(proxies.available())                                   # ['weather']
print(proxies("weather/current/Tokyo", {}))
print(proxies("weather/forecast/daily", {"city": "Tokyo", "days": 2}))
print(proxies.retrieve_api_docs("weather"))                  # what query_api_doc returns
```

```text
## Weather API (weather)
Current weather and forecasts for any city.
- **current/{city}** [GET] – Current temperature and conditions for a city.
    - `city*` (str) e.g. London
    - `units` (str) e.g. celsius
- **forecast/daily** [GET] – Daily forecast for a city.
    - `city*` (str) e.g. London
    - `days` (int) e.g. 5
```

`AgentInterpreter` is the interpreter behind `run_python`. Its namespace
persists across calls, like a live REPL:

```python
interpreter = AgentInterpreter(proxies, max_output_chars=5000, timeout=60)

print(interpreter.run('tokyo = CALL_API("weather/current/Tokyo", {})\nprint(tokyo["temperature"])'))
print(interpreter.run('print("Tokyo is", tokyo["condition"])'))   # `tokyo` is still defined
print(interpreter.run("1 / 0").splitlines()[-1])                  # ZeroDivisionError: division by zero
```

- Only what the code prints is returned; the last expression is not echoed.
- Exceptions come back as formatted tracebacks, so the agent can fix its code
  without losing state.
- Output longer than `max_output_chars` is cut and `"... (truncated)"` appended.
- Code running longer than `timeout` seconds raises `TimeoutError`, which the
  tool loop reports to the model as an error.
- `interpreter.reset()` clears everything except `CALL_API`.

### Step 14: Give an agent the proxy

Add a `proxy` block to an agent's configuration. LLLM then builds a
`ProxyManager` and an `AgentInterpreter` for the agent, appends the API
directory and usage instructions to its system prompt, and adds the
`query_api_doc` and `run_python` tools, which are offered on every turn:

```python
from lllm import Tactic


class WeatherTactic(Tactic):
    name = "weather"
    agent_group = ["weather_analyst"]

    def call(self, task: str) -> str:
        analyst = self.agents["weather_analyst"]
        analyst.open("main")
        analyst.receive(task)
        return analyst.respond().content


config = {
    "global": {"model_name": "gpt-4o", "model_args": {"temperature": 0.1}},
    "agent_configs": [{
        "name": "weather_analyst",
        "system_prompt": "You are a weather analyst. Use the Weather API to answer questions.",
        "proxy": {
            "activate_proxies": ["weather"],
            "exec_env": "interpreter",
            "max_output_chars": 5000,
            "timeout": 60.0,
        },
    }],
}

tactic = WeatherTactic(config)
print(sorted(tactic.agents["weather_analyst"].system_prompt.functions))
# ['query_api_doc', 'run_python']
```

The same configuration in YAML, with a global default that agents override
field by field:

```yaml
global:
  model_name: gpt-4o
  proxy:
    activate_proxies: [weather]
    exec_env: interpreter

agent_configs:
  - name: weather_analyst
    system_prompt: You are a weather analyst. Use the Weather API to answer questions.

  - name: quick_analyst
    system_prompt: You answer weather questions in one sentence.
    proxy:
      timeout: 30.0          # only this field changes; the rest is inherited
```

Every key of the `proxy` block:

```yaml
proxy:
  activate_proxies: [weather]      # proxies to load; empty means all registered proxies
  deploy_mode: false               # passed to each proxy instance
  cutoff_date: "2024-01-01"        # ISO date passed to proxies that restrict data by date
  exec_env: interpreter            # "interpreter" adds run_python; any other value (or null) does not
  max_output_chars: 5000           # truncate run_python output; 0 disables truncation
  truncation_indicator: "... (truncated)"
  timeout: 60.0                    # seconds per run_python call
  prompt_template: null            # custom instruction block; null selects the built-in one
```

With `exec_env` set to anything other than `interpreter` (for example
`jupyter` or `null`), the agent still gets the API directory and
`query_api_doc`, but no `run_python`: your tactic executes code itself, as in
Part 3.

The built-in instruction block includes the API directory: every endpoint of
every activated proxy, with its parameters. For large APIs that is a lot of
system-prompt text. To keep it short, set `prompt_template` to your own
instructions without the `{api_directory}` placeholder; the agent then reads
documentation only through `query_api_doc`. A custom template may use
`{max_output_chars}` and `{truncation_indicator}`, and must double any literal
braces. Set it before building the tactic:

```python
config["agent_configs"][0]["proxy"]["prompt_template"] = (
    "\n\nCall query_api_doc(name) before using a data API, then run Python with "
    "run_python, calling CALL_API(path, params). Output longer than "
    "{max_output_chars} characters is truncated."
)
tactic = WeatherTactic(config)
```

### Step 15: Run it

```python
print(tactic("What is the current weather in Tokyo and Paris?"))
```

The agent typically:

1. calls `query_api_doc("weather")` to read the endpoint details;
2. calls `run_python` with code that runs `CALL_API("weather/current/Tokyo", {})`
   and `CALL_API("weather/current/Paris", {})` and prints the results;
3. calls `run_python` again, reusing those variables, to compare or format;
4. writes the final answer.

Each tactic call builds fresh agents, so every call starts with a fresh
interpreter.

LLLM also ships ready-made proxies for research data, registered by
`load_builtin_proxies()` from `lllm.runtimes.native.proxies`: Financial
Modeling Prep (`fmp`), FRED (`fred`), Exa search (`exa`), Google Trends (`gt`),
Wolfram Alpha (`wa`), and Main Street Data (`msd`). Each reads its API key from
the environment. [Lesson 8](08-advanced-patterns.md) combines several of them.

---

## Part 3: Sandbox

Interpreter mode runs the agent's code for you. When you want to own execution,
to intercept every cell, keep a notebook-style record, or run code somewhere
else, let the agent write code cells and run them in your tactic.

### Step 16: Let the agent write cells and run them yourself

The agent writes `<python_cell>` and `<markdown_cell>` tags; a parser extracts
them; the tactic runs the code in a sandbox and sends the outputs back:

```python
import re

from lllm import ProxyManager, Prompt, Tactic
from lllm.runtimes.native import ParseError
from lllm.runtimes.native.proxies import AgentInterpreter

NOTEBOOK_SYSTEM = """You are a data analyst working in a Python notebook.

- Write code in <python_cell>...</python_cell> tags and notes in <markdown_cell>...</markdown_cell> tags.
- Cells run in order and variables persist between turns. Use print() to see values.
- Call the data APIs listed below with CALL_API(path, params).
- When you are done, write a final <markdown_cell> summary followed by <TERMINATE_NOTEBOOK>.

{api_directory}"""

CELL = re.compile(r"<(python_cell|markdown_cell)>(.*?)</\1>", re.DOTALL)


def notebook_parser(content: str, **kwargs) -> dict:
    cells = [(kind, body.strip()) for kind, body in CELL.findall(content)]
    has_code = any(kind == "python_cell" for kind, _ in cells)
    terminate = "<TERMINATE_NOTEBOOK>" in content and not has_code
    if not cells and not terminate:
        raise ParseError(
            "No cells found. Wrap code in <python_cell>...</python_cell>, "
            "or reply with <TERMINATE_NOTEBOOK> when you are done."
        )
    return {"cells": cells, "terminate": terminate}


task_turn = Prompt(path="notebook/task", prompt="{task}", parser=notebook_parser)
results_turn = Prompt(path="notebook/results", prompt="Cell outputs:\n\n{outputs}", parser=notebook_parser)


class NotebookAnalysisTactic(Tactic):
    name = "notebook_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, max_steps: int = 8) -> str:
        proxies = ProxyManager(activate_proxies=["weather"])
        sandbox = AgentInterpreter(proxies, timeout=60)   # CALL_API is pre-loaded
        notebook = []

        analyst = self.agents["analyst"]
        analyst.open("main", prompt_args={"api_directory": proxies.retrieve_api_docs()})
        analyst.receive_prompt(task_turn, {"task": task})

        for _ in range(max_steps):
            parsed = analyst.respond().parsed
            outputs = []
            for kind, source in parsed["cells"]:
                notebook.append(f"<{kind}>\n{source}\n</{kind}>")
                if kind == "python_cell":
                    try:
                        output = sandbox.run(source)
                    except Exception as exc:          # e.g. a timeout
                        output = f"{type(exc).__name__}: {exc}"
                    outputs.append(output or "(no output)")
            if parsed["terminate"]:
                break
            analyst.receive_prompt(results_turn, {"outputs": "\n---\n".join(outputs) or "(no code cells)"})
        return "\n\n".join(notebook)


config = {
    "global": {"model_name": "gpt-4o", "model_args": {"temperature": 0.1}},
    "agent_configs": [{"name": "analyst", "system_prompt": NOTEBOOK_SYSTEM}],
}

print(NotebookAnalysisTactic(config)("Compare the 3-day forecast for Tokyo and Paris."))
```

Notice how the pieces from earlier lessons fit together:

- Both turn prompts carry `notebook_parser`, because every turn's reply must be
  parsed (the rule from Lesson 3).
- A reply without cells fails parsing, so the model is asked again with the
  `ParseError` message before your loop ever sees it.
- A failing cell returns its traceback as output, so the agent sees the error
  on the next turn and can rewrite the cell.

### Step 17: Bring your own sandbox

The tactic only needs an object with `run(code) -> str`. Swap the sandbox and
nothing else changes. This one runs each cell in a fresh Python process:

```python
import subprocess
import sys


class SubprocessSandbox:
    """Run each cell in a fresh Python process: no shared state, no CALL_API."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def run(self, code: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.stdout + result.stderr


print(SubprocessSandbox().run("print(sum(range(10)))"))   # 45
```

Use it by replacing `sandbox = AgentInterpreter(...)` in the tactic. The same
shape fits a container, a remote execution service, or a database query runner.

`AgentInterpreter` runs code with `exec()` inside your process. That suits
trusted analysis code calling your own APIs, but it is not isolation. For
untrusted code, put real isolation (a container, VM, or remote sandbox) behind
the `run()` method.

---

## Summary

| Concept | API |
|---|---|
| Define a tool | `@tool(description=..., prop_desc={...})` |
| Call a tool directly | `fn(FunctionCall(name=..., arguments={...}))` |
| Agent tools (every turn) | system prompt `function_list=[...]`, or config `tools: [...]` |
| Tools for one turn | `Prompt(function_list=[...])` + `agent.receive_prompt(prompt, args)` |
| Turn without agent tools | `Prompt(..., agent_tools=False)` |
| Cap tool rounds | `Agent(max_interrupt_steps=N)` |
| Inspect tool rounds | `agent.respond(return_session=True).interrupts` |
| Custom result text | `@tool(processor=fn)` |
| Declaration + implementation | `Function(name=..., properties=...)` + `runtime.register_tool(name, impl)` |
| Tactic as a tool | `tactic_as_function(tactic)` |
| MCP server | `Prompt(mcp_servers_list=[MCP(...)])` |
| Define a proxy | `@ProxyRegistrator(path=...)` + `@BaseProxy.endpoint(endpoint="a/{b}", ...)` |
| HTTP-backed proxy | `base_url`, `api_key_name`, `api_key`; endpoint methods return the params |
| Use a proxy in Python | `ProxyManager(activate_proxies=[...])("proxy/path", params)`, `AgentInterpreter(manager)` |
| Give an agent proxies | config `proxy: {activate_proxies: [...], exec_env: interpreter}` |
| Own code execution | `<python_cell>` parser + any object with `run(code) -> str` |

See the [Proxies and Sandbox reference](../core/proxy-and-sandbox.md) and the
[Prompts reference](../core/prompts.md) for every option.

**Next:** [5. Native Tactics](05-native-tactics.md)
