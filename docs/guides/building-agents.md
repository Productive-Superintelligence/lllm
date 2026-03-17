# Tutorial: Build a Full Package

This tutorial builds a complete LLLM package step by step — starting with a single agent and ending with a multi-agent system that has structured configs, session logging, and clear extension points. Each step is runnable on its own; later steps build on the structure introduced earlier.

By the end you'll have a project that looks like this:

```
research_writer/
├── lllm.toml
├── prompts/
│   ├── researcher_system.md
│   └── writer_system.md
├── configs/
│   └── research_writer.yaml
├── tactics/
│   └── research_writer.py
└── main.py
```

---

## Step 1: Single agent, one question

The simplest possible setup — no files, no config:

```python
from lllm import Tactic

agent = Tactic.quick("You are a helpful assistant.", model="gpt-4o")
agent.open("session")
agent.receive("What is the capital of France?")
print(agent.respond().content)
```

`Tactic.quick()` returns a bare `Agent`. The `open / receive / respond` pattern maps to: start a conversation, add a message, get a reply.

---

## Step 2: Multi-turn conversation

Reuse the same dialog alias to continue a conversation:

```python
from lllm import Tactic

agent = Tactic.quick("You are a helpful assistant.", model="gpt-4o")
agent.open("chat")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    agent.receive(user_input)
    reply = agent.respond()
    print(f"Agent: {reply.content}")
```

The dialog accumulates messages across turns. Each call to `respond()` sees the full history.

---

## Step 3: Structured output

Use a Pydantic model as the output format:

```python
from pydantic import BaseModel
from lllm import Tactic
from lllm.core.prompt import Prompt

class Summary(BaseModel):
    headline: str
    key_points: list[str]
    sentiment: str

prompt = Prompt(
    path="summarizer/system",
    prompt="You are a document summarizer. Return structured JSON.",
    format=Summary,
)

agent = Tactic.quick(prompt, model="gpt-4o")
agent.open("work")
agent.receive("Article text goes here...")
result = agent.respond()
summary: Summary = result.parsed   # typed Pydantic object
print(summary.headline)
```

---

## Step 4: Adding tools

Link Python functions as tools the LLM can call:

```python
from lllm import Tactic
from lllm.core.prompt import Prompt, Function

def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"

weather_fn = Function.from_callable(
    get_weather,
    description="Get the current weather for a city",
)

prompt = Prompt(
    path="assistant/system",
    prompt="You are a helpful assistant with access to weather data.",
    functions=[weather_fn],
)

agent = Tactic.quick(prompt, model="gpt-4o")
agent.open("session")
agent.receive("What's the weather like in Tokyo?")
print(agent.respond().content)
```

The agent call loop automatically executes tool calls, collects results, and continues until the model produces a final text response.

---

## Step 5: Organize as a Package

Once you have multiple prompts or agents, move from a single file to a package. This is when `lllm.toml` enters the picture.

Create the folder structure:

```
research_writer/
├── lllm.toml
├── prompts/
│   ├── researcher_system.md
│   └── writer_system.md
└── main.py
```

**`lllm.toml`** — declares the package and its resource folders:

```toml
[package]
name = "research_writer"
version = "0.1.0"

[prompts]
paths = ["prompts/"]

[configs]
paths = ["configs/"]

[tactics]
paths = ["tactics/"]
```

**`prompts/researcher_system.md`** — a system prompt as a plain Markdown file:

```
You are a research analyst. Given a topic, provide a thorough analysis
with key findings, evidence, and open questions.

Topic: {topic}
```

LLLM scans the `prompts/` folder at startup and registers every `.md` file and every `Prompt` object in `.py` files automatically. No import or registration code needed.

Load a prompt by name:

```python
from lllm import load_prompt

prompt = load_prompt("researcher_system")   # resolves to research_writer.prompts:researcher_system
```

All resources are namespaced under the package name declared in `lllm.toml`. See [Package System](../architecture/packages.md) for the full reference on namespacing, dependencies, and aliasing.

---

## Step 6: Multi-agent tactic

Add an agent config YAML and a `Tactic` subclass:

```
research_writer/
├── lllm.toml
├── prompts/
│   ├── researcher_system.md
│   └── writer_system.md
├── configs/
│   └── research_writer.yaml      ← new
└── tactics/
    └── research_writer.py         ← new
```

**`configs/research_writer.yaml`** — describes the agents declaratively:

```yaml
agent_group_configs:
  researcher:
    model_name: gpt-4o
    system_prompt_path: researcher_system
    temperature: 0.3
    max_completion_tokens: 4000
  writer:
    model_name: gpt-4o
    system_prompt_path: writer_system
    temperature: 0.7
```

**`tactics/research_writer.py`** — orchestrates the agents:

```python
from lllm import Tactic

class ResearchWriter(Tactic):
    name = "research_writer"
    agent_group = ["researcher", "writer"]

    def call(self, topic: str, **kwargs) -> str:
        researcher = self.agents["researcher"]
        writer = self.agents["writer"]

        researcher.open("research", prompt_args={"topic": topic})
        findings = researcher.respond()

        writer.open("draft", prompt_args={"findings": findings.content, "topic": topic})
        return writer.respond().content
```

Run it:

```python
from lllm import build_tactic, resolve_config

config = resolve_config("research_writer")
tactic = build_tactic(config, ckpt_dir="./runs")
result = tactic("The impact of quantum computing on cryptography")
print(result)
```

`build_tactic` discovers `ResearchWriter` from the `tactics/` folder (auto-registered on import during discovery), resolves the agent configs, and returns a ready-to-call tactic instance.

---

## Step 7: Batch and async execution

Run the same tactic over many inputs:

```python
topics = [
    "Quantum computing",
    "Large language models",
    "Neuromorphic chips",
]

# Sequential
results = [tactic(t) for t in topics]

# Parallel (thread pool)
results = tactic.bcall(topics, max_workers=3)

# Parallel with partial failure tolerance
results = tactic.bcall(topics, max_workers=3, fail_fast=False)
# items that failed are returned as Exception objects

# Async streaming (yields results as they complete, with original index)
async for idx, result in tactic.ccall(topics):
    print(f"[{idx}] {result}")
```

---

## Step 8: Session logging

Attach a log store to track costs, inputs, outputs, and traces:

```python
from lllm import build_tactic, resolve_config
from lllm.logging import sqlite_store

store = sqlite_store("./logs.db", partition="experiments")

config = resolve_config("research_writer")
tactic = build_tactic(config, ckpt_dir="./runs", log_store=store)

result = tactic(
    "Quantum computing",
    tags={"experiment": "baseline", "split": "test"},
)
```

Every call is automatically saved under the stable key `research_writer::research_writer`. Query sessions later:

```python
summaries = store.list_sessions(tactic_path="research_writer")
for s in summaries:
    print(s.session_id, f"${s.total_cost:.4f}", s.state)

# Filter by tags
baseline = store.list_sessions(tags={"experiment": "baseline"})

# Drill into a session
session = store.load_session(summaries[0].session_id)
for agent_name, calls in session.agent_sessions.items():
    for call in calls:
        print(f"  {agent_name}: {call.state}  cost={call.cost}")
```

See [Logging](../core/logging.md) for the full query API, tag system, and cost reports.

---

## Advanced Customization

The package system gives you the full structure. These are the deep extension points when you need to go further.

### Custom Invoker

Swap or extend the LLM backend by subclassing `BaseInvoker`:

```python
from lllm.invokers.base import BaseInvoker, InvokeResult
from lllm.core.dialog import Dialog

class MyInvoker(BaseInvoker):
    def call(self, dialog: Dialog, model: str, model_args=None, **kwargs) -> InvokeResult:
        # call your own backend, mock API, or add tracing
        ...
        return InvokeResult(message=message)

# Pass to build_tactic
tactic = build_tactic(config, ckpt_dir="./runs", invoker=MyInvoker())
```

See [Invokers](../core/invokers.md) for the full interface, streaming support, and LiteLLM details.

### Custom Log Backend

Connect any storage system by subclassing `LogBackend`:

```python
from lllm.logging import LogBackend, LogStore

class RedisBackend(LogBackend):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...
    def delete(self, key: str) -> None: ...

store = LogStore(RedisBackend(redis.Redis()), partition="prod")
```

See [Logging](../core/logging.md) for Redis and Firestore backend examples, design notes, and the full `LogBackend` interface.

### Custom Proxy Tool

Build a structured tool system for agents by subclassing `BaseProxy`:

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="my_tool/search", name="Search API", description="...")
class SearchProxy(BaseProxy):
    @BaseProxy.endpoint(
        category="web",
        endpoint="query",
        description="Search the web",
        params={"q*": (str, "query string")},
        response={"results": ["..."]},
    )
    def search(self, params): ...
```

See [Proxy & Tools](../core/proxy-and-sandbox.md) for the proxy system documentation.

---

## What You've Built

At this point you have a complete LLLM package:

```
research_writer/
├── lllm.toml          ← package manifest + resource discovery
├── prompts/           ← .md files auto-registered as prompts
├── configs/           ← .yaml files auto-registered as agent configs
├── tactics/           ← Tactic subclasses auto-registered
└── main.py
```

This package can be shared, imported as a dependency by another package, and its tactics can be called by higher-level systems without modification.

---

## Next Steps

- [Package System](../architecture/packages.md) — namespacing, dependencies, aliasing, custom sections
- [Configuration](../core/config.md) — `lllm.toml`, YAML config inheritance, and `vendor_config`
- [Prompts](../core/prompts.md) — templates, parsers, tools, and handlers in depth
- [Agent](../core/agent.md) — how the call loop handles errors and interrupts
- [Tactics](../core/tactic.md) — sub-tactics, typed I/O, and registration
- [Project Reference](project-template.md) — naming conventions and folder layout
