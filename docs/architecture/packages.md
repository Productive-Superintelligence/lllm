# Package System

After the quick start, you have one script and one agent. That works fine for experiments. But as your system grows — multiple prompts, multiple agents, tactics you want to reuse across projects — you need a way to organise and discover resources without hardcoding paths everywhere.

The LLLM **package system** is that organisational layer. A package is a folder with an `lllm.toml` manifest that declares where your prompts, configs, tactics, and proxies live. LLLM handles discovery, namespacing, and lazy loading from there.

This is what the jump looks like:

```python
# Stage 1: everything inline, paths hardcoded
agent = Tactic.quick("You are a research assistant.", model="gpt-4o")

# Stage 3: resources discovered from a package — no paths in Python code
config = resolve_config("research_writer")   # reads configs/research_writer.yaml
tactic = build_tactic(config, ckpt_dir="./runs")
result = tactic("Analyze transformer architectures")
```

The package system makes this possible without changing how your `Tactic` or `Prompt` code is written. You add `lllm.toml` and the folder structure; LLLM wires everything together.

For a step-by-step walkthrough of building a complete package from scratch, see [Tutorial: Build a Full Package](../guides/building-agents.md).

---

LLLM is Pythonic: you can use it purely as a library and organise your own code however you want. But separating LLLM resources (prompts, proxies, configs, tactics) into a package is the recommended approach for anything beyond a single script.

A package is structured as:

```
package_name/
  ├── prompts/        # Prompt objects for agents to call
  ├── proxies/        # Proxy-based tool implementations
  ├── configs/        # Agent configuration YAML files
  ├── tactics/        # Tactic subclasses
  └── lllm.toml       # Package metadata and resource declarations
```

A typical project layout:

```
project_name/
├── lllm.toml           # root package — LLLM finds this automatically
├── lllm_packages/
│   ├── my_package1/
│   └── my_package2/
└── ... (application code, data, etc.)
```

Conceptually, LLLM maintains a registry of prompts, proxies, configs, and tactics, loaded on demand through the `lllm.toml` file. All resources are indexed by URLs of the form `<package>.<section>:<resource_path>`. Tactics are the top-level building blocks — they find agent configs by key, prompts from the prompt registry, and proxy-based tools from the proxy registry.


## Runtime Initialization

Everything is automatic. When `lllm` is imported, it searches upward from `cwd` for `lllm.toml` and loads the full package tree into the default runtime. `load_prompt("my_prompt")` works immediately with no setup code.

If no `lllm.toml` is found, LLLM falls back to scanning the current directory for any of the four standard folders (`prompts/`, `configs/`, `tactics/`, `proxies/`). If found, those are loaded and a `RuntimeWarning` recommends adding an `lllm.toml`. If neither exists, the runtime starts empty (fast mode).

```python
from lllm import load_prompt  # runtime already populated at import time
```

For advanced use (testing with isolated registries, parallel experiments), you can load additional named runtimes explicitly — see [Runtime](../core/runtime.md).


## Resources

LLLM has four built-in resource types: prompts, proxies, configs, and tactics. You can also define custom resource types via custom TOML sections.

Every resource is internally wrapped in a `ResourceNode` object, which manages the qualified key, namespace, lazy loading, and metadata. `ResourceNode` is a **wrapper**, not a base class — the existing classes (Prompt, Tactic, BaseProxy) do not inherit from it.

For eager resources (prompts, tactics discovered at import time), the value is set immediately. For lazy resources (config YAML files, custom assets), the `ResourceNode` holds a loader callable and the file is only read on first access, then cached.


## lllm.toml Format

An `lllm.toml` has six official sections: `[package]`, `[prompts]`, `[proxies]`, `[configs]`, `[tactics]`, and `[dependencies]`. Custom sections like `[assets]` are also supported.

- **[package]**: Package identity — name, version, description. All resources declared in this TOML are namespaced under this package name.
- **[prompts]**: Paths to prompt folders. Defaults to `prompts/` if omitted. Empty if neither the section nor the subfolder exists.
- **[proxies]**: Paths to proxy folders. Defaults to `proxies/`.
- **[configs]**: Paths to config folders (YAML files, loaded lazily). Defaults to `configs/`.
- **[tactics]**: Paths to tactic folders. Defaults to `tactics/`.
- **[dependencies]**: Paths to other packages. Dependencies are loaded into their own namespace only. To re-export a dependency's resources into the current package namespace, list their paths explicitly in the relevant resource section.


## Resource Indexing

Resources are indexed by URLs with the format `<package_name>.<section_name>:<resource_path>`. There is always exactly one `:` separator.

The `resource_path` is built from the folder structure relative to the declared path root: `<subfolder>/.../<filename>/<object_name>`. Root folders are stripped — multiple `paths` entries merge into a flat namespace.

When a key collision occurs during discovery (two paths producing the same resource key), LLLM logs a warning. The later registration overwrites the earlier one. Use the `under` keyword to disambiguate.

### Example

Given `[prompts] paths = [".../prompts_1", ".../prompts_2"]` under package `my_pkg`:

```
prompts_1/
├── greet.py          # contains: hello, goodbye
├── sub/
    ├── deep.py       # contains: analyzer

prompts_2/
├── tools.py          # contains: searcher
```

The resulting URLs are:

- `my_pkg.prompts:greet/hello`
- `my_pkg.prompts:greet/goodbye`
- `my_pkg.prompts:sub/deep/analyzer`
- `my_pkg.prompts:tools/searcher`

### Convenience Access

Full URL via `load_resource` (always requires section):

```python
load_resource("my_pkg.prompts:greet/hello")
load_resource("prompts:greet/hello")          # section-only → default package
```

Typed convenience functions (section inferred):

```python
load_prompt("my_pkg:greet/hello")             # package-qualified
load_prompt("greet/hello")                     # bare → default package namespace
```


## Resource Loading

### Dependency-Only Loading

```toml
[package]
name = "my_system"

[dependencies]
packages = ["./packages/child_pkg", "../shared/shared_pkg"]
```

Each dependency's resources live in their own namespace:

```python
load_prompt("child_pkg:greet/hello")
load_prompt("shared_pkg:tools/searcher")
```

No resources exist under `my_system` — dependencies are isolated unless re-exported.


### Re-Exporting Into the Current Namespace

List paths explicitly in resource sections to include them in the current package's namespace:

```toml
[package]
name = "my_system"

[prompts]
paths = ["./packages/child_pkg/prompts", "../shared/shared_pkg/prompts"]

[dependencies]
packages = ["./packages/child_pkg", "../shared/shared_pkg"]
```

Now resources are accessible via both namespaces:

```python
load_prompt("greet/hello")                    # via my_system (default)
load_prompt("child_pkg:greet/hello")          # via dependency namespace
```


## Alias Loading

Two mechanisms address name collisions:

- **`as`** on dependencies — creates an additional alias. The original name still works.
- **`under`** on resource paths — adds a virtual root folder prefix within the importing package's namespace.

Both can be specified as inline strings or standard TOML inline tables — the two forms are exactly equivalent:

```toml
# String keyword form
packages = ["./packages/child_pkg as cp"]
paths = ["./some/path under vendor"]

# TOML inline table form
packages = [{path = "./packages/child_pkg", alias = "cp"}]
paths = [{path = "./some/path", prefix = "vendor"}]

# Table form also accepts the keyword names
packages = [{path = "./packages/child_pkg", as = "cp"}]
paths = [{path = "./some/path", under = "vendor"}]
```

### Example

```toml
[package]
name = "my_system"

[tactics]
paths = ["./pkg1/tactics under v1", "./pkg2/tactics under v2"]

[prompts]
paths = ["./pkg1/prompts", "./pkg2/prompts under vendor"]

[dependencies]
packages = ["./pkg1 as p1", "./pkg2 as p2"]
```

Access patterns:

```python
# Tactics from pkg1, re-exported into my_system with v1 prefix:
load_tactic("v1/my_tactic")                 # via default namespace
load_tactic("my_system:v1/my_tactic")        # explicit

# Same tactic via pkg1's own namespace (no prefix):
load_tactic("p1:my_tactic")                  # via alias
load_tactic("pkg1:my_tactic")                # via original name (still works)

# Prompts from pkg2 with vendor prefix:
load_prompt("vendor/my_prompt")              # via default namespace
load_prompt("p2:my_prompt")                  # via alias (no prefix)
```

Note: `under` modifies how resources appear in the **importing** package's namespace, not in the source package's own namespace.


## Custom Sections

Beyond the four built-in sections, you can define any custom section in `lllm.toml` to package arbitrary files — images, ML model weights, JSON schemas, data files, or anything else your system needs.

### How It Works

Custom sections follow the same `paths` / `under` mechanics as built-in sections. During discovery, LLLM walks the declared folders and registers every file as a lazy `ResourceNode`. Files are **not read until first access** — a package with 500MB of model weights costs nothing at import time.

File loading behavior depends on extension:

| Extension | Loaded as |
| --- | --- |
| `.json` | Parsed via `json.load` → `dict` / `list` |
| `.yaml`, `.yml` | Parsed via `yaml.safe_load` → `dict` / `list` |
| `.toml` | Parsed via `tomllib.load` → `dict` |
| Everything else | Raw `bytes` via `Path.read_bytes()` |

Resource keys **include the file extension** (unlike Python-based sections where `.py` is stripped), because the extension is part of the file identity — `logo.png` and `logo.svg` are different resources.

Any `.py` files in custom section folders are also scanned for Python-defined resources (Prompt, Tactic, BaseProxy subclasses), so you can mix code and data in the same section.

### Declaring Custom Sections

```toml
[package]
name = "my_toolkit"

[assets]
paths = ["assets"]

[models]
paths = ["models"]

[schemas]
paths = ["schemas"]
```

With this directory structure:

```
my_toolkit/
├── lllm.toml
├── assets/
│   ├── logo.png
│   ├── banner.svg
│   └── templates/
│       └── email.html
├── models/
│   └── classifier.pt
└── schemas/
    └── api_spec.json
```

### Accessing Custom Resources

Use `load_resource` with `"pkg.section:path"` or `"section:path"` (section-only uses default package):

```python
from lllm import load_resource

# Full URL
logo_bytes = load_resource("my_toolkit.assets:logo.png")           # → bytes
api_spec = load_resource("my_toolkit.schemas:api_spec.json")       # → dict (parsed)

# Section-only (if my_toolkit is the default package)
logo_bytes = load_resource("assets:logo.png")
html = load_resource("assets:templates/email.html")                # → bytes

# Nested paths work naturally
weights = load_resource("models:classifier.pt")                    # → bytes
```

### Getting the File Path Directly

For large files or custom formats where the default loader isn't appropriate (e.g., loading a PyTorch model with `torch.load`), access the `ResourceNode` directly to get the file path:

```python
from lllm import get_default_runtime

runtime = get_default_runtime()
node = runtime.get_node("my_toolkit.models:classifier.pt")

# The absolute file path is stored in metadata
file_path = node.metadata["file_path"]

# Use your own loader
import torch
model = torch.load(file_path)
```

### Custom Sections with `under` Prefix

The `under` keyword works the same way as for built-in sections:

```toml
[assets]
paths = [
    "./icons under ui",
    "./photos under content",
]
```

```python
load_resource("assets:ui/check.svg")
load_resource("assets:content/hero.jpg")
```


## Resource Access Reference

This section covers how each resource type is registered, discovered, and accessed. All resource types share the same URL scheme (`pkg.section:key`) and the same `ResourceNode` infrastructure, but they differ in how they enter the registry and how you typically use them.


### Prompts

Prompts are Python `Prompt` objects defined at module scope in `.py` files.

**Registration paths:**

1. **Discovery** (recommended) — list folders in `[prompts]` section of `lllm.toml`. Every `Prompt` instance found at module scope is registered automatically.
2. **Manual** — `runtime.register_prompt(prompt, namespace="pkg.prompts")` or the module-level `register_prompt(prompt)`.

**Defining a prompt:**

```python
# prompts/research.py
from lllm import Prompt

system = Prompt(
    path="system",
    prompt="You are a research analyst. Analyze {topic}.",
)

followup = Prompt(
    path="followup",
    prompt="Based on the analysis, suggest next steps for {topic}.",
)
```

With `[prompts] paths = ["prompts"]` under package `my_pkg`, these register as:
- `my_pkg.prompts:research/system`
- `my_pkg.prompts:research/followup`

**Accessing:**

```python
from lllm import load_prompt

# Bare key (default package)
prompt = load_prompt("research/system")

# Package-qualified
prompt = load_prompt("my_pkg:research/system")

# Full URL
prompt = load_prompt("my_pkg.prompts:research/system")
```

**In agent configs** — reference by path:

```yaml
agent_configs:
  - name: analyst
    system_prompt_path: research/system      # bare key
    # or: system_prompt_path: my_pkg:research/system
```


### Proxies

Proxies are `BaseProxy` subclasses decorated with `@ProxyRegistrator`. They have their own dispatch system (`Proxy` runtime) on top of the resource registry.

**Registration paths:**

1. **`@ProxyRegistrator` decorator** (primary) — registers the class into the default runtime at import time. The decorator sets `_proxy_path` on the class, which becomes the resource key.

2. **Discovery** — list folders in `[proxies]` section of `lllm.toml`. Discovery imports the `.py` files, which triggers `@ProxyRegistrator` decorators.

3. **`load_builtin_proxies()`** — manually imports LLLM's bundled proxy modules to trigger their decorators. Use this in notebooks or scripts without an `lllm.toml`.

**Defining a proxy:**

```python
# proxies/weather.py
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(
    path="weather/openweather",
    name="OpenWeather API",
    description="Current weather and forecasts",
)
class OpenWeatherProxy(BaseProxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("OPENWEATHER_API_KEY")

    @BaseProxy.endpoint(
        category="current",
        endpoint="weather",
        description="Get current weather for a city",
        params={"q*": (str, "London"), "units": (str, "metric")},
        response={"temp": 15.2, "description": "cloudy"},
    )
    def current_weather(self, params):
        return params
```

**Loading and using proxies:**

```python
# Via the Proxy runtime dispatcher (recommended for agent use)
from lllm.proxies import Proxy

proxy = Proxy(activate_proxies=["weather/openweather"])
result = proxy("weather/openweather/current/weather", {"q": "London"})
```

See [Proxy & Tools](../core/proxy-and-sandbox.md) for full proxy documentation.


### Tactics

Tactics are `Tactic` subclasses that register automatically when defined (via `__init_subclass__`).

**Registration paths:**

1. **Auto-registration** (primary) — defining a `Tactic` subclass with a `name` attribute registers it at class definition time (import time).

2. **Discovery** — list folders in `[tactics]` section of `lllm.toml`. Discovery imports `.py` files, triggering auto-registration.

3. **Manual** — `register_tactic_class(MyTactic, runtime=my_runtime)`.

**Defining a tactic:**

```python
# tactics/research.py
from lllm import Tactic

class ResearchTactic(Tactic):
    name = "researcher"
    agent_group = ["analyst", "searcher"]

    def call(self, task: str, **kwargs) -> str:
        analyst = self.agents["analyst"]
        analyst.open("work", prompt_args={"topic": task})
        return analyst.respond().content
```

**Loading and using:**

```python
from lllm import build_tactic, resolve_config

config = resolve_config("default")
tactic = build_tactic(config, ckpt_dir="./runs", name="researcher")
result = tactic("Analyze transformer architectures")
```


### Configs

Configs are YAML files discovered from `[configs]` folders. They are loaded **lazily** — the file is only read on first access.

**Accessing:**

```python
from lllm import load_config, resolve_config

# Direct load (no inheritance resolution)
cfg = load_config("default")

# With base inheritance resolution (recommended)
cfg = resolve_config("experiments/fast")
```

See [Configuration](../core/config.md) for full details on `global` merge, `base` inheritance, and `vendor_config`.


### Custom Sections

Any TOML section besides the six official ones is treated as a custom resource section. All non-Python files are registered lazily; Python files are also scanned for typed resources.

See the [Custom Sections](#custom-sections) section above for full details.
