# Context & Registries

Every LLLM runtime needs a place to look up prompts, proxies, and agent types by name. `Context` is that place — a lightweight container that holds the three registries and the discovery flag. It replaces the scattered module-level dicts that previously served this role.

## Why Context Exists

Before Context, registration state lived in three independent globals: `PROMPT_REGISTRY`, `PROXY_REGISTRY`, and `AGENT_REGISTRY`. This worked for simple scripts but caused problems in practice:

- **Testing** required manually resetting each dict between test cases, and missing one led to cross-test pollution.
- **Parallel experiments** (e.g., comparing two prompt sets in the same process) were impossible — there was only one global namespace.
- **Import order mattered.** Registries were populated as side effects of importing modules, making the system fragile and hard to reason about.

Context consolidates these into a single object with a clean lifecycle.

## The Default Context

Most users never interact with Context directly. A default instance is created when `lllm.core.context` is imported — no scanning, no side effects, just an empty container:

```python
from lllm.core.context import get_default_context

ctx = get_default_context()
print(ctx.prompts)   # {} — empty until something registers
print(ctx.proxies)   # {}
print(ctx.agents)    # {}
```

Every convenience function in the framework (`register_prompt`, `register_proxy`, `register_agent_class`, `auto_discover`) operates on this default instance when no explicit context is passed. The rule is simple: **if you don't pass a context, you get the global one.**

## Registering Into a Context

### Prompts

```python
from lllm import Prompt, register_prompt

# Module-level convenience — registers into default context
my_prompt = Prompt(path="chat/greeting", prompt="Hello, {name}!")
register_prompt(my_prompt)
```

Auto-discovery does the same thing. When `lllm.toml` lists a prompts folder, every `Prompt` object found at module scope gets registered into the context that discovery was invoked with.

### Proxies

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="search/web", name="Web Search", description="Search the web")
class WebProxy(BaseProxy):
    ...
```

`ProxyRegistrator` registers the class into the context at decoration time. Like prompts, this goes into the default context unless an explicit context is passed.

### Agent Types

Agent type registration is automatic. Subclassing `Orchestrator` triggers `__init_subclass__`, which calls `register_agent_class`:

```python
from lllm import Orchestrator

class MyAgent(Orchestrator):
    agent_type = "my_agent"
    agent_group = ["assistant"]
    
    def call(self, task, **kwargs):
        ...
# MyAgent is now registered as "my_agent" in the default context
```

Because `__init_subclass__` fires at class definition time (before any instance exists), agent types always register into the default context. Agent *instances*, however, use whatever context is passed to `Orchestrator.__init__`.

## How Context Flows Through the System

The context is threaded as an optional parameter that defaults to the global instance. This means existing code never has to change, but advanced users can inject their own.

**Orchestrator** receives and stores it:

```python
agent = MyAgent(config, ckpt_dir="./ckpt", context=my_context)
# agent._context is now my_context
# all prompt lookups, discovery, and agent construction use my_context
```

**Discovery** populates it:

```python
from lllm.core.discovery import auto_discover

auto_discover(context=my_context)
# scans lllm.toml folders, registers prompts and proxies into my_context
```

**Proxy runtime** reads from it:

```python
from lllm.proxies import Proxy

proxy = Proxy(activate_proxies=["search/web"], context=my_context)
# instantiates only proxies registered in my_context
```

**Dialog.from_dict** uses it to resolve prompt references when reconstructing a dialog from a checkpoint:

```python
dialog = Dialog.from_dict(saved_data, log_base=log, context=my_context)
```

## Isolated Contexts for Testing

The primary use case for explicit contexts is test isolation. Each test can create a fresh context, register what it needs, and tear down without affecting anything else:

```python
def test_my_agent():
    ctx = Context()
    ctx.register_prompt(Prompt(path="test/prompt", prompt="..."))
    
    agent = MyAgent(config, ckpt_dir="/tmp/test", context=ctx)
    result = agent("hello")
    
    assert "test/prompt" in ctx.prompts
    # no cleanup needed — ctx is garbage collected
```

For test suites that share a context across multiple tests, `ctx.reset()` clears all registries and resets the discovery flag:

```python
@pytest.fixture(autouse=True)
def clean_context():
    ctx = get_default_context()
    yield
    ctx.reset()
```

## Isolated Contexts for Parallel Experiments

Researchers can run two agent configurations side-by-side in the same process by giving each its own context:

```python
from lllm.core.context import Context
from lllm.core.discovery import auto_discover

ctx_a = Context()
ctx_b = Context()

# Populate with different prompt sets
auto_discover(config_path="experiment_a/lllm.toml", context=ctx_a)
auto_discover(config_path="experiment_b/lllm.toml", context=ctx_b)

agent_a = MyAgent(config_a, ckpt_dir="./run_a", context=ctx_a)
agent_b = MyAgent(config_b, ckpt_dir="./run_b", context=ctx_b)
```

Each agent sees only the prompts and proxies registered in its own context. Discovery runs independently for each (tracked by `ctx._discovery_done`).

## API Reference

### `Context`

| Method | Description |
| --- | --- |
| `register_prompt(prompt, overwrite=True)` | Add a `Prompt` keyed by `prompt.path`. Raises `ValueError` on duplicate if `overwrite=False`. |
| `get_prompt(path)` | Retrieve a prompt by path. Raises `KeyError` if not found. |
| `register_proxy(name, proxy_cls, overwrite=False)` | Add a `BaseProxy` subclass keyed by name. |
| `register_agent(agent_type, agent_cls, overwrite=False)` | Add an `Orchestrator` subclass keyed by agent type string. |
| `reset()` | Clear all registries and reset the discovery flag. |

### Module-Level Helpers

| Function | Description |
| --- | --- |
| `get_default_context()` | Returns the process-wide default `Context` instance. |
| `set_default_context(ctx)` | Replace the default instance. Use sparingly — mainly for test harnesses or application bootstrap. |

## Design Notes

- **Creating a Context has zero side effects.** No file scanning, no imports, no network calls. It is an empty container until something explicitly registers into it.
- **One piece of global state remains:** the `_DEFAULT_AUTO_DISCOVER` flag in `discovery.py`, which controls whether `auto_discover_if_enabled` runs by default. This is a process-wide behavioral preference, not registry state, so it deliberately lives outside Context.
- **Context is not thread-safe.** If you need concurrent registration (unusual in practice), wrap calls in a lock or create per-thread contexts. The normal usage — discover once at startup, read many times during agent calls — has no contention.