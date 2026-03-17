# Runtime & Registries

!!! note "Most users never touch this"
    The runtime initialises automatically when you `import lllm`. You only need this page if you are writing tests with isolated registries, running parallel experiments with separate runtimes, or integrating LLLM into a larger system that manages its own package loading.

Every LLLM runtime is a unified `ResourceNode`-based store keyed by namespaced URLs. It's where prompts, configs, tactics, and proxies are looked up by name.


## Automatic Initialization

`import lllm` triggers `_auto_init()`, which:

1. Searches upward from `cwd` for `lllm.toml` (or uses `$LLLM_CONFIG`).
2. If found — loads the full package tree (dependencies, all resource sections).
3. If not found — scans `cwd` for any of the standard folders (`prompts/`, `configs/`, `tactics/`, `proxies/`). If any exist, loads them and emits a `RuntimeWarning` suggesting you add an `lllm.toml`. If none exist, starts empty (fast mode, no output).

This means **you never call `load_runtime()` in normal usage** — it has already run by the time your first line of application code executes.


## The Default Runtime

A default instance is created when `lllm.core.runtime` is imported — empty, no side effects. `_auto_init()` populates it on first import.

```python
from lllm.core.runtime import get_default_runtime

runtime = get_default_runtime()
print(runtime.keys("prompt"))   # list qualified keys of all prompts
```


## Named Runtimes

For parallel experiments or testing, create named runtimes:

```python
from lllm import load_runtime, get_runtime

load_runtime("experiment_a/lllm.toml", name="exp_a")
load_runtime("experiment_b/lllm.toml", name="exp_b")

rt_a = get_runtime("exp_a")
rt_b = get_runtime("exp_b")
```

Each runtime has its own isolated registry. `load_runtime()` with no name replaces the default.


## Internal Architecture

All resources are stored as `ResourceNode` objects in a single `_resources` dict, keyed by their qualified key (e.g. `"my_pkg.prompts:greet/hello"`). A `_type_index` maps resource types to sets of qualified keys for fast filtered iteration.

```
Runtime
├── _resources: Dict[str, ResourceNode]    # "pkg.section:key" → node
├── _type_index: Dict[str, Set[str]]       # "prompt" → {"pkg.prompts:a", ...}
├── packages: Dict[str, PackageInfo]       # "pkg_name" → metadata
├── _loaded_package_paths: Set[str]        # cycle detection
└── _default_namespace: Optional[str]      # root package name
```


## Resource Resolution

When you call `runtime.get(key)` or any typed convenience method, the resolution order is:

1. **Exact match** — if `key` exists in `_resources`, return it.
2. **Default namespace fallback** — if `key` has no `:`, try `default_ns.<section>s:key` for each built-in section (prompts, proxies, tactics, configs).
3. **Section injection** — if `key` has `:` but no `.` in the package part (e.g. `"pkg:foo"`), and a `resource_type` is specified, try `"pkg.<type>s:foo"`.

This means:
- `get_prompt("greet/hello")` → tries `my_system.prompts:greet/hello`
- `get_prompt("child_pkg:greet/hello")` → tries `child_pkg.prompts:greet/hello`
- `get("my_pkg.prompts:greet/hello")` → exact match


## Registering Resources

### Via Discovery (Automatic)

`load_package()` scans folders declared in `lllm.toml` and registers everything it finds:

```python
from lllm import load_package

load_package("path/to/lllm.toml")
# All prompts, proxies, tactics, configs now in the runtime
```

### Via Typed Convenience Methods

```python
from lllm import Prompt
from lllm.core.runtime import get_default_runtime

runtime = get_default_runtime()

# Register a prompt
p = Prompt(path="test/greeting", prompt="Hello {name}!")
runtime.register_prompt(p, namespace="my_pkg.prompts")

# Register a proxy class
runtime.register_proxy("search/web", WebProxy, namespace="my_pkg.proxies")

# Register a tactic class
runtime.register_tactic("analytica", AnalyticaTactic, namespace="my_pkg.tactics")

# Register a config (eager or lazy)
runtime.register_config("default", {"model_name": "gpt-4o"}, namespace="my_pkg.configs")
runtime.register_config("heavy", loader=lambda: yaml.safe_load(open("heavy.yaml")),
                         namespace="my_pkg.configs")
```

### Via Low-Level API

```python
from lllm.core.resource import ResourceNode

node = ResourceNode.eager("logo.png", image_bytes,
                          namespace="my_pkg.assets", resource_type="asset")
runtime.register(node)

# Lazy loading
node = ResourceNode.lazy("big_data", loader=lambda: load_parquet("data.parquet"),
                          namespace="my_pkg.datasets", resource_type="dataset")
runtime.register(node)
```


## Retrieving Resources

```python
# Typed convenience (section inferred)
prompt = runtime.get_prompt("greet/hello")
tactic_cls = runtime.get_tactic("analytica")
proxy_cls = runtime.get_proxy("search/web")
config = runtime.get_config("default")

# Generic (requires full URL or section+key)
value = runtime.get("my_pkg.assets:logo.png")

# Module-level convenience
from lllm import load_prompt, load_tactic, load_config, load_resource

prompt = load_prompt("child_pkg:greet/hello")
config = load_config("experiments/ablation")
asset = load_resource("my_pkg.assets:logo.png")
```


## Isolated Runtimes for Testing

```python
from lllm.core.runtime import Runtime

def test_my_agent():
    runtime = Runtime()
    runtime.register_prompt(Prompt(path="test/prompt", prompt="..."),
                            namespace="test.prompts")
    agent = MyAgent(config, ckpt_dir="/tmp/test", runtime=runtime)
    result = agent("hello")
    assert runtime.has("test.prompts:test/prompt")
```

`runtime.reset()` clears all registries and resets state:

```python
@pytest.fixture(autouse=True)
def clean_runtime():
    runtime = get_default_runtime()
    yield
    runtime.reset()
```


## API Reference

### `Runtime`

| Method | Description |
| --- | --- |
| `register(node, overwrite=True)` | Register any `ResourceNode`. |
| `get(key, resource_type=None)` | Retrieve a resource value with namespace resolution. |
| `get_node(key, resource_type=None)` | Retrieve the `ResourceNode` itself (not `.value`). |
| `has(key)` | Check existence without raising. |
| `keys(resource_type=None)` | List qualified keys, optionally filtered. |
| `register_prompt(prompt, overwrite, namespace)` | Typed registration for prompts. |
| `get_prompt(path)` | Typed retrieval for prompts. |
| `register_proxy(name, cls, overwrite, namespace)` | Typed registration for proxies. |
| `get_proxy(path)` | Typed retrieval for proxies. |
| `register_tactic(name, cls, overwrite, namespace)` | Typed registration for tactics. |
| `get_tactic(name)` | Typed retrieval for tactics. |
| `register_config(name, data, overwrite, namespace, loader)` | Typed registration (eager or lazy). |
| `get_config(path)` | Typed retrieval (triggers lazy load). |
| `register_package(pkg)` | Register a `PackageInfo`. |
| `reset()` | Clear all state. |

### Module-Level Helpers

| Function | Description |
| --- | --- |
| `get_default_runtime()` | Returns the process-wide default `Runtime`. |
| `set_default_runtime(rt)` | Replace the default. |
| `get_runtime(name=None)` | Get a runtime by name (`None` = default). |
| `load_runtime(path=None, name=None)` | Create, populate, and store a runtime. |


## Design Notes

- **Creating a Runtime has zero side effects.** No scanning, no imports. It is an empty container until something registers into it.
- **Single storage, smart resolution.** Each resource is stored once under its qualified key. Bare-key and package-shorthand access is handled at read time by `_resolve()`, not by storing duplicates.
- **Runtime is not thread-safe** for concurrent *registration*. The normal pattern — discover once at startup, read many times during agent calls — has no contention. For concurrent registration, wrap in a lock or use per-thread runtimes.