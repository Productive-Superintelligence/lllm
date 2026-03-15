# Runtime & Registries

Every LLLM runtime needs a place to look up prompts, proxies, and agent types by name. `Runtime` is that place — a lightweight container that holds the three registries and the discovery flag. It replaces the scattered module-level dicts that previously served this role.

## The Default Runtime

Most users never interact with Runtime directly. A default instance is created when `lllm.core.runtime` is imported — no scanning, no side effects, just an empty container:

```python
from lllm.core.runtime import get_default_runtime

runtime = get_default_runtime()
print(runtime.prompts)   # {} — empty until something registers
print(runtime.proxies)   # {}
print(runtime.agents)    # {}
```

Every convenience function in the framework (`register_prompt`, `register_proxy`, `register_agent_class`, `auto_discover`) operates on this default instance when no explicit runtime is passed. The rule is simple: **if you don't pass a runtime, you get the global one.**

## Registering Into a Runtime

### Prompts

```python
from lllm import Prompt, register_prompt

# Module-level convenience — registers into default runtime
my_prompt = Prompt(path="chat/greeting", prompt="Hello, {name}!")
register_prompt(my_prompt)
```

Auto-discovery does the same thing. When `lllm.toml` lists a prompts folder, every `Prompt` object found at module scope gets registered into the runtime that discovery was invoked with.

### Proxies

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="search/web", name="Web Search", description="Search the web")
class WebProxy(BaseProxy):
    ...
```

`ProxyRegistrator` registers the class into the runtime at decoration time. Like prompts, this goes into the default runtime unless an explicit runtime is passed.

### Agent Types

Agent type registration is automatic. Subclassing `Orchestra` triggers `__init_subclass__`, which calls `register_agent_class`:

```python
from lllm import Orchestra

class MyAgent(Orchestra):
    agent_type = "my_agent"
    agent_group = ["assistant"]
    
    def call(self, task, **kwargs):
        ...
# MyAgent is now registered as "my_agent" in the default runtime
```

Because `__init_subclass__` fires at class definition time (before any instance exists), agent types always register into the default runtime. Agent *instances*, however, use whatever runtime is passed to `Orchestra.__init__`.

## How Context Flows Through the System

The runtime is threaded as an optional parameter that defaults to the global instance. This means existing code never has to change, but advanced users can inject their own.

**Orchestra** receives and stores it:

```python
agent = MyAgent(config, ckpt_dir="./ckpt", runtime=my_runtime)
# agent._runtime is now my_runtime
# all prompt lookups, discovery, and agent construction use my_runtime
```

**Discovery** populates it:

```python
from lllm.core.discovery import auto_discover

auto_discover(runtime=my_runtime)
# scans lllm.toml folders, registers prompts and proxies into my_runtime
```

**Proxy runtime** reads from it:

```python
from lllm.proxies import Proxy

proxy = Proxy(activate_proxies=["search/web"], runtime=my_runtime)
# instantiates only proxies registered in my_runtime
```

**Dialog.from_dict** uses it to resolve prompt references when reconstructing a dialog from a checkpoint:

```python
dialog = Dialog.from_dict(saved_data, log_base=log, runtime=my_runtime)
```

## Isolated Runtimes for Testing

The primary use case for explicit runtimes is test isolation. Each test can create a fresh runtime, register what it needs, and tear down without affecting anything else:

```python
def test_my_agent():
    runtime = Runtime()
    runtime.register_prompt(Prompt(path="test/prompt", prompt="..."))
    
    agent = MyAgent(config, ckpt_dir="/tmp/test", runtime=runtime)
    result = agent("hello")
    
    assert "test/prompt" in runtime.prompts
    # no cleanup needed — ctx is garbage collected
```

For test suites that share a runtime across multiple tests, `runtime.reset()` clears all registries and resets the discovery flag:

```python
@pytest.fixture(autouse=True)
def clean_runtime():
    runtime = get_default_runtime()
    yield
    runtime.reset()
```

## Isolated Runtimes for Parallel Experiments

Researchers can run two agent configurations side-by-side in the same process by giving each its own runtime:

```python
from lllm.core.runtime import Runtime
from lllm.core.discovery import auto_discover

runtime_a = Runtime()
runtime_b = Runtime()

# Populate with different prompt sets
auto_discover(config_path="experiment_a/lllm.toml", runtime=runtime_a)
auto_discover(config_path="experiment_b/lllm.toml", runtime=runtime_b)

agent_a = MyAgent(config_a, ckpt_dir="./run_a", runtime=runtime_a)
agent_b = MyAgent(config_b, ckpt_dir="./run_b", runtime=runtime_b)
```

Each agent sees only the prompts and proxies registered in its own runtime. Discovery runs independently for each (tracked by `runtime._discovery_done`). 

## API Reference

### `Runtime`

| Method | Description |
| --- | --- |
| `register_prompt(prompt, overwrite=True)` | Add a `Prompt` keyed by `prompt.path`. Raises `ValueError` on duplicate if `overwrite=False`. |
| `get_prompt(path)` | Retrieve a prompt by path. Raises `KeyError` if not found. |
| `register_proxy(name, proxy_cls, overwrite=False)` | Add a `BaseProxy` subclass keyed by name. |
| `register_agent(agent_type, agent_cls, overwrite=False)` | Add an `Orchestra` subclass keyed by agent type string. |
| `reset()` | Clear all registries and reset the discovery flag. |

### Module-Level Helpers

| Function | Description |
| --- | --- |
| `get_default_runtime()` | Returns the process-wide default `Runtime` instance. |
| `set_default_runtime(runtime)` | Replace the default instance. Use sparingly — mainly for test harnesses or application bootstrap. |

## Design Notes

- **Creating a Runtime has zero side effects.** No file scanning, no imports, no network calls. It is an empty container until something explicitly registers into it.
- **One piece of global state remains:** the `_DEFAULT_AUTO_DISCOVER` flag in `discovery.py`, which controls whether `auto_discover_if_enabled` runs by default. This is a process-wide behavioral preference, not registry state, so it deliberately lives outside Runtime.
- **Runtime is not thread-safe.** If you need concurrent registration (unusual in practice), wrap calls in a lock or create per-thread runtimes. The normal usage — discover once at startup, read many times during agent calls — has no contention.