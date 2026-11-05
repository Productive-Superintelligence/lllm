# Package Discovery

A single script with one agent is fine for experiments. As a system grows
(several prompts, several agents, tactics you want to reuse across projects)
you need a way to organise and find resources without hardcoding import paths
everywhere.

The native **package system** is that organisational layer. A package is a
folder with an `lllm.toml` manifest that declares where its prompts, tools,
proxies, configs, and tactics live. LLLM discovers them, namespaces them, and
loads data lazily.

The jump looks like this:

```python
# Stage 1: everything inline
from lllm import Tactic

agent = Tactic.quick(system_prompt="You are a research assistant.", model="gpt-4o")

# Stage 3: resources discovered from a package, no paths in Python code
from lllm import build_tactic, load_runtime, resolve_config

load_runtime()                                   # reads ./lllm.toml
tactic = build_tactic(resolve_config("research_writer"))
result = tactic("Analyze transformer architectures")
```

Your `Tactic` and `Prompt` code does not change; you add `lllm.toml` and the
folder structure, and LLLM wires everything together. You can always use LLLM
purely as a library, but separating resources into a package is the recommended
shape for anything beyond a single script.

---

## Package layout

```
package_name/
├── lllm.toml       # package metadata and resource declarations
├── prompts/        # Prompt objects
├── tools/          # @tool Function implementations
├── proxies/        # BaseProxy subclasses
├── configs/        # agent config YAML files
├── tactics/        # Tactic subclasses
├── assets/         # data files (optional)
└── services/       # ASGI apps and service spec files (optional)
```

`lllm create my-app` generates a working package in this shape (`lllm.toml`,
`prompts/`, `configs/default.yaml`, `tactics/`, a `main.py` that calls
`load_runtime()`, and an offline smoke test). Pick another starting point with
`--template pipeline|proxy|research|service`, or point `--template` at a folder
containing `lllm-template.toml`.

A typical project:

```
project_name/
├── lllm.toml               # root package
├── lllm_packages/          # drop-in third-party packages (auto-loaded)
│   ├── finance-toolkit/
│   └── data-utils/
└── ...                     # application code, data, etc.
```

Conceptually, LLLM keeps one registry of resources, indexed by URLs of the form
`<package>.<section>:<resource_path>`. Tactics find agent configs by key, prompts
in the prompt registry, tools in the tool registry, and proxies in the proxy
registry.

---

## Loading packages

Loading is explicit. Call `load_runtime()` at startup:

```python
from lllm import load_prompt, load_runtime

load_runtime()                          # or load_runtime("path/to/lllm.toml")
prompt = load_prompt("research/system")
```

`load_runtime()` uses `$LLLM_CONFIG` if set, otherwise the nearest `lllm.toml`
(or `.lllm.toml`, `LLLM.toml`) in the working directory or its parents. It loads
that package and its dependency tree into a fresh runtime and makes it the
default.

If there is no manifest, it scans the working directory for the standard
folders (`prompts/`, `tools/`, `proxies/`, `configs/`, `tactics/`, `assets/`),
loads any it finds as a package named after the directory, and emits a
`RuntimeWarning` recommending an `lllm.toml`. If none exist, the runtime stays
empty. Pass `discover_cwd=False` to disable the scan.

After the main package, `load_runtime()` also loads **shared packages** from the
standard drop-in folders; see [Sharing packages](#sharing-packages). For
isolated or named runtimes, see [Runtime & Registries](../core/runtime.md).

---

## Resources

There are seven built-in resource types (prompts, tools, proxies, configs,
tactics, assets, services), and custom sections add your own.

Every resource is wrapped in a `ResourceNode` that manages its qualified key,
namespace, lazy loading, and metadata. `ResourceNode` is a wrapper, not a base
class: `Prompt`, `Function`, `Tactic`, and `BaseProxy` do not inherit from it.

Python resources (prompts, tools, proxies, tactics, service apps) are
registered eagerly when discovery imports their modules. File resources
(configs, assets, service specs, custom sections, data files in `tools/`) are
registered lazily: the node holds a loader and the file is read on first access,
then cached.

---

## `lllm.toml` for native discovery

```toml
[package]
name = "my_system"
version = "0.1.0"
description = "Research assistant"

[prompts]
paths = ["prompts"]

[tools]
paths = ["tools"]

[proxies]
paths = ["proxies"]

[configs]
paths = ["configs"]

[tactics]
paths = ["tactics"]

[assets]
paths = ["assets"]

[services]
paths = ["services"]

[dependencies]
packages = ["./packages/child_pkg"]
```

- **`[package]`**: identity. All resources from this manifest are namespaced
  under `name` (default: the folder name).
- **`[prompts]`, `[tools]`, `[proxies]`, `[configs]`, `[tactics]`, `[assets]`,
  `[services]`**:
  `paths` lists folders relative to the manifest. Without `paths`, the
  same-named subfolder is used if it exists; otherwise the section is empty. A
  listed path that does not exist is skipped with a `RuntimeWarning`.
- **`[dependencies]`**: `packages` lists other package folders. Each loads into
  its own namespace only.
- **Any other table with `paths`** is a [custom section](#custom-sections).

The same manifest can also carry hub metadata (`org`, `kind`, named resources
such as `[tactics.brief]` with an `entry`). Native discovery reads only the keys
above and ignores the hub-declared sections (`schemas`, `channels`,
`snapshots`, `runs`, `config`, `docs`, `examples`, `requirements`, `card`), so
one file serves both:

```toml
[package]
name = "research-kit"
org = "acme"
version = "0.2.0"

[prompts]
paths = ["prompts"]

[tactics]
paths = ["tactics"]          # native discovery

[tactics.brief]              # hub resource
entry = "research_kit.tactics:build_tactic"
runtime = "native"
```

See [Package Manifest](../packages/manifest.md) and [Hub](../packages/hub.md)
for the hub side.

---

## Resource URLs and keys

Resources are indexed by `<package_name>.<section_name>:<resource_path>`, with
exactly one `:`. The resource path depends on the type:

| Section | Resource path |
| --- | --- |
| `prompts` | `<file path without .py>/<prompt.path>` |
| `tools` | `<file path>` when the tool's name equals the file name, else `<file path>/<tool name>`; other files: `<file path with extension>` |
| `proxies` | The `@ProxyRegistrator` path, else `<file path>/<ClassName>` |
| `tactics` | `Tactic.name` (the file path is not part of the key) |
| `configs` | `<file path without .yaml/.yml>` |
| `services` | Module-level ASGI apps (FastAPI/Starlette): `<file path>/<attribute>`; spec files (`.json`, `.yaml`, `.toml`): `<file path with extension>`, loaded as data |
| `assets`, custom sections | `<file path with extension>` |

File paths are relative to the declared folder, so the folder itself is
stripped, and several `paths` entries merge into one flat namespace. An `under`
prefix is prepended to every key from that entry. Files whose names start with
`_` (and data files starting with `.`) are skipped, as is `__pycache__`.

When two entries produce the same key, the later one wins (logged at debug
level). Use `under` to keep them apart.

### Example

Given `[prompts] paths = ["prompts_1", "prompts_2"]` in package `my_pkg`:

```
prompts_1/
├── greet.py          # Prompt objects with path "hello" and "goodbye"
└── sub/
    └── deep.py       # path "analyzer"

prompts_2/
└── tools.py          # path "searcher"
```

The URLs are:

- `my_pkg.prompts:greet/hello`
- `my_pkg.prompts:greet/goodbye`
- `my_pkg.prompts:sub/deep/analyzer`
- `my_pkg.prompts:tools/searcher`

### Convenience access

`load_resource` takes a URL with a section:

```python
from lllm import load_prompt, load_resource

load_resource("my_pkg.prompts:greet/hello")
load_resource("prompts:greet/hello")          # section only: default package
```

Typed helpers infer the section:

```python
load_prompt("my_pkg:greet/hello")             # package-qualified
load_prompt("greet/hello")                    # bare: default package
load_prompt("my_pkg.prompts:greet/hello")     # full URL
```

The default package is the first one loaded into the runtime (the root
package). Package shorthand (`pkg:key`) works for every typed helper, including
`load_proxy("my_pkg:weather")`.

---

## Resource loading

### Dependency-only loading

```toml
[package]
name = "my_system"

[dependencies]
packages = ["./packages/child_pkg", "../shared/shared_pkg"]
```

Each dependency's resources live in its own namespace:

```python
load_prompt("child_pkg:greet/hello")
load_prompt("shared_pkg:tools/searcher")
```

Nothing is registered under `my_system` unless you re-export it. Dependencies
load recursively, each package folder is loaded once (cycles are detected), and
a dependency folder without a manifest is skipped with a `RuntimeWarning`.

### Re-exporting into the current namespace

List a dependency's folders in your own sections to include them in your
namespace as well:

```toml
[package]
name = "my_system"

[prompts]
paths = ["./packages/child_pkg/prompts", "../shared/shared_pkg/prompts"]

[dependencies]
packages = ["./packages/child_pkg", "../shared/shared_pkg"]
```

```python
load_prompt("greet/hello")             # via my_system (the default package)
load_prompt("child_pkg:greet/hello")   # via the dependency's namespace
```

---

## Aliases

Two mechanisms resolve name collisions:

- **`as`** on dependencies registers an additional namespace for the package. The
  original name keeps working.
- **`under`** on resource paths adds a virtual folder prefix inside the
  importing package's namespace.

Both have a string form and an equivalent TOML table form:

```toml
# String keyword form
packages = ["./packages/child_pkg as cp"]
paths = ["./some/path under vendor"]

# Table form
packages = [{path = "./packages/child_pkg", alias = "cp"}]
paths = [{path = "./some/path", prefix = "vendor"}]

# The table form also accepts the keyword names
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

```python
from lllm import load_prompt, load_tactic

# Tactics from pkg1, re-exported into my_system with the v1 prefix
load_tactic("v1/my_tactic")              # default namespace
load_tactic("my_system:v1/my_tactic")    # explicit

# The same tactic through pkg1's own namespace (no prefix)
load_tactic("p1:my_tactic")              # alias
load_tactic("pkg1:my_tactic")            # original name

# Prompts from pkg2 with the vendor prefix
load_prompt("vendor/my_file/my_prompt")  # default namespace
load_prompt("p2:my_file/my_prompt")      # alias, no prefix
```

`under` changes how resources appear in the **importing** package's namespace,
not in the source package's own namespace. A re-exported module is imported
once per namespace it appears in.

---

## Custom sections

Beyond the built-in sections, any table with `paths` packages arbitrary files:
images, model weights, JSON schemas, datasets.

### How it works

Custom sections use the same `paths` and `under` rules. Discovery walks the
folders and registers every file as a lazy `ResourceNode`; nothing is read until
first access, so a package with 500 MB of weights costs nothing at load time.

| Extension | Loaded as |
| --- | --- |
| `.json` | `json.load` → `dict` / `list` |
| `.yaml`, `.yml` | `yaml.safe_load` → `dict` / `list` |
| `.toml` | `tomllib.load` → `dict` |
| anything else | raw `bytes` |

Keys **include the file extension** (unlike Python sections, where `.py` is
dropped), because `logo.png` and `logo.svg` are different resources. The
`[assets]` section behaves the same way and also has a typed helper,
`load_asset`.

Keep Python modules in the built-in sections (`prompts/`, `tools/`, `proxies/`,
`tactics/`); use custom sections for data files.

### Declaring custom sections

```toml
[package]
name = "my_toolkit"

[models]
paths = ["models"]

[datasets]
paths = ["datasets"]
```

```
my_toolkit/
├── lllm.toml
├── assets/
│   ├── logo.png
│   └── templates/
│       └── email.html
├── models/
│   └── classifier.pt
└── datasets/
    └── labels.json
```

### Accessing custom resources

```python
from lllm import load_resource
from lllm.runtimes.native.core.resource import load_asset

logo = load_resource("my_toolkit.assets:logo.png")          # bytes
labels = load_resource("my_toolkit.datasets:labels.json")   # parsed JSON

# Section only, when my_toolkit is the default package
html = load_resource("assets:templates/email.html")        # bytes
weights = load_resource("models:classifier.pt")            # bytes

logo = load_asset("my_toolkit:logo.png")
```

### Getting the file path

For large files or formats the default loader does not suit (a PyTorch
checkpoint, say), take the path from the node:

```python
from lllm import get_default_runtime

node = get_default_runtime().get_node("my_toolkit.models:classifier.pt")
file_path = node.metadata["file_path"]      # absolute path

import torch
model = torch.load(file_path)
```

### `under` prefixes

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

---

## Resource access reference

All types share the URL scheme and the `ResourceNode` store; they differ in how
they enter the registry and how you use them.

### Prompts

Python `Prompt` objects at module scope.

Registration:

1. **Discovery**: folders in `[prompts]`; every module-level `Prompt` is
   registered.
2. **Manually**: `runtime.register_prompt(prompt, namespace="pkg.prompts")`, or
   `register_prompt(prompt)` for the default runtime.

```python
# prompts/research.py
from lllm import Prompt

system = Prompt(path="system", prompt="You are a research analyst. Analyze {topic}.")
followup = Prompt(path="followup", prompt="Suggest next steps for {topic}.")
```

With `[prompts] paths = ["prompts"]` in package `my_pkg`, these register as
`my_pkg.prompts:research/system` and `my_pkg.prompts:research/followup`.

```python
from lllm import load_prompt

prompt = load_prompt("research/system")               # bare (default package)
prompt = load_prompt("my_pkg:research/system")        # package-qualified
prompt = load_prompt("my_pkg.prompts:research/system")  # full URL
```

In agent configs, reference prompts by path:

```yaml
agent_configs:
  - name: analyst
    system_prompt_path: research/system      # or my_pkg:research/system
```

### Tools

`Function` objects, usually created with `@tool`. Two package-friendly styles:

- **Coupled**: the decorated function is both declaration and implementation.
  Import it in local code, or reference its URL from a prompt or config.
- **Decoupled**: a prompt declares a `Function` schema and the package provides an
  `@tool(name=...)` implementation registered under exactly that name, like a
  header file plus an implementation file.

Registration:

1. **Discovery**: folders in `[tools]`; every module-level `Function` is
   registered. Other files in those folders are registered as lazy data
   resources.
2. **Manually**: `runtime.register_tool("name", fn, namespace="pkg.tools")`.

```python
# tools/search.py
from lllm import tool


@tool(name="search", description="Search internal documents.")
def search(query: str, top_k: int = 5) -> str:
    return f"Top {top_k} results for {query}"
```

With `[tools] paths = ["tools"]` in package `my_pkg`, this registers as
`my_pkg.tools:search`, because the tool's name matches its file. A module with
several tools nests them under the module path: `tools/math.py` with `square`
and `cube` gives `my_pkg.tools:math/square` and `my_pkg.tools:math/cube`.

A decoupled declaration binds to the tool whose key is exactly its name, so give
each such implementation its own module named after it (`tools/search.py`) or
register it explicitly.

```python
from lllm import Prompt

Prompt(
    path="research/system",
    prompt="Use tools when needed.",
    function_list=["my_pkg.tools:search"],
)
```

In agent configs, `tools` can mix regular tools, tactic tools, and proxies:

```yaml
global:
  tools:
    - my_pkg.tools:search
    - shared_pkg.tactics:code_review
    - market_pkg.proxies:prices
```

### Proxies

`BaseProxy` subclasses, usually decorated with `@ProxyRegistrator`, dispatched
at run time by `ProxyManager`.

Registration:

1. **`@ProxyRegistrator`**: registers the class into the default runtime at
   import time (or the runtime you pass) and sets `_proxy_path`, the bare key.
2. **Discovery**: folders in `[proxies]`; importing the modules registers each
   `BaseProxy` subclass as `pkg.proxies:<_proxy_path>`.
3. **`load_builtin_proxies()`**: imports LLLM's bundled proxy modules.

```python
# proxies/weather.py
import os

from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(path="weather", name="OpenWeather API", description="Current weather and forecasts")
class OpenWeatherProxy(BaseProxy):
    base_url = "https://api.openweathermap.org/data/2.5"
    api_key_name = "appid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("OPENWEATHER_API_KEY")

    @BaseProxy.endpoint(
        category="current",
        endpoint="weather",
        description="Get current weather for a city",
        params={"q*": (str, "London"), "units": (str, "metric")},
        response={"main": {"temp": 15.2}, "weather": [{"description": "cloudy"}]},
    )
    def current_weather(self, params: dict) -> dict:
        params.setdefault("units", "metric")
        return params
```

```python
from lllm import ProxyManager, load_proxy

proxy_cls = load_proxy("my_pkg:weather")
proxy = ProxyManager(activate_proxies=["weather"])
result = proxy("weather/weather", {"q": "London"})   # GET <base_url>/weather?q=London&units=metric&appid=...
```

See [Proxies & Sandbox](../core/proxy-and-sandbox.md) for the full proxy
system.

### Tactics

`Tactic` subclasses with a `name`.

Registration:

1. **Discovery**: folders in `[tactics]`; every `Tactic` subclass with a `name`
   is registered under that name.
2. **At class definition**: `class MyTactic(Tactic, runtime=registry)`.
3. **Manually**: `register_tactic_class(MyTactic, runtime=registry)`.

```python
# tactics/research.py
from lllm import Tactic


class ResearchTactic(Tactic):
    name = "researcher"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs) -> str:
        analyst = self.agents["analyst"]
        analyst.open("work", prompt_args={"topic": task})
        return analyst.respond().content
```

```python
from lllm import build_tactic, resolve_config

tactic = build_tactic(resolve_config("default"), name="researcher")
result = tactic("Analyze transformer architectures")
```

Tactics built this way record the stable identity `my_pkg::researcher` on their
sessions; see [Logging](../core/logging.md#tactic-identity-the-stable-pkgname-key).

### Configs

YAML files from `[configs]` folders, loaded lazily.

```python
from lllm import load_config, resolve_config

cfg = load_config("default")                 # no inheritance
cfg = resolve_config("experiments/fast")     # with base inheritance (recommended)
```

See [Configuration](../core/config.md) for `global` merging, `base` inheritance,
and `vendor_config`.

### Services

Files from `[services]` folders. Module-level ASGI apps (FastAPI or Starlette
instances; names starting with `_` are skipped) register as
`pkg.services:<file path>/<attribute>`, and service spec files (`.json`,
`.yaml`, `.toml`) as `pkg.services:<file path with extension>`, loaded as data:

```python
from lllm.runtimes.native.core.resource import load_service_ref

app = load_service_ref("my_pkg:app/app")        # services/app.py, attribute `app`
spec = load_service_ref("my_pkg:api.yaml")      # services/api.yaml, parsed
```

How services are launched is covered in [Services](../protocol/services.md).

### Assets and custom sections

Files from `[assets]` or any custom section, loaded lazily by extension; see
[Custom sections](#custom-sections).

---

## Sharing packages

A package is a folder with an `lllm.toml`, so sharing one is sharing that
folder: through git, a zip, or anything else. The consumer drops it into a
standard location and it is loaded with no config changes.

### Drop-in folders

`load_runtime()` loads every package found in:

| Folder | Scope |
| --- | --- |
| `<project root>/lllm_packages/<pkg>/` | Project: shared with everyone using the repo |
| `~/.lllm/packages/<pkg>/` | User: available in all your projects |

The project root is the folder containing the project's `lllm.toml` (or the
working directory if there is none). Each sub-folder must contain an
`lllm.toml`; it is loaded exactly like an explicit dependency, into its own
namespace. A package that fails to load produces a `RuntimeWarning` instead of
stopping startup.

To use a published package:

```bash
git clone https://github.com/acme/finance-toolkit lllm_packages/finance-toolkit
# or
unzip finance-toolkit-v1.2.zip -d lllm_packages/
```

On the next `load_runtime()` it is live:

```python
from lllm import load_prompt, resolve_config

prompt = load_prompt("finance-toolkit:analysis/system")
config = resolve_config("finance-toolkit:research")
```

Commit `lllm_packages/` so the team gets the same packages; keep
`~/.lllm/packages/` for personal ones.

### Installing, exporting, listing, removing

```python
from lllm import export_package, install_package, list_packages, remove_package

# Export a loaded package to a zip (a top-level folder named after the package)
export_package("finance-toolkit", "dist/finance-toolkit-1.2.zip")
export_package("finance-toolkit", "dist/finance-toolkit-1.2-full.zip", bundle_deps=True)

# Install (user scope by default) and load it into the current runtime
install_package("dist/finance-toolkit-1.2.zip")
install_package("dist/finance-toolkit-1.2.zip", alias="ft", scope="project")

for pkg in list_packages():
    print(pkg["name"], pkg["version"], pkg["scope"], pkg["path"])

remove_package("ft", scope="project")
```

| Scope | Location | Visibility |
| --- | --- | --- |
| `user` (default) | `~/.lllm/packages/<name>/` | All your projects |
| `project` | `<project root>/lllm_packages/<name>/` | This repo (commit it) |

- `install_package` expects a zip with `lllm.toml` at the top level or in exactly
  one top-level folder. It refuses to overwrite an installed package
  (`FileExistsError`); install under another name with `alias`, which renames
  the folder and rewrites `name` in its `lllm.toml` (lowercase letters, digits,
  and hyphens). `load=False` installs without loading.
- `export_package` skips hidden files and `__pycache__`. With
  `bundle_deps=True`, every transitive dependency from `[dependencies]` is
  packed under `lllm_packages/` inside the zip, so recipients get everything in
  one install. `.zip` is appended to the output path if it has another suffix.
- `remove_package` deletes the installed folder and unregisters the package;
  with no `scope` it raises `ValueError` if the name exists in both scopes.

### Explicit dependency

For tighter control (pinning a path, aliasing the namespace, re-exporting into
your own namespace) declare the package as a dependency:

```toml
[dependencies]
packages = [
    "./lllm_packages/finance-toolkit",
    "./lllm_packages/data-utils as du",
]
```

Explicit and drop-in loading can be combined. A package folder that appears in
both is loaded once.

### Packages vs. skills

Both packages and [agent skills](../core/agent.md#skills) are shareable,
drop-in, version-controlled folders. Choose by what you are sharing:

| | Agent skills | LLLM packages |
| --- | --- | --- |
| **Contains** | `SKILL.md` instructions, scripts, reference files | Prompts, tactics, tools, proxies, agent configs |
| **Drop-in path** | `.agents/skills/<name>/` | `lllm_packages/<name>/` |
| **Consumed by** | Any agentskills-compatible agent | LLLM |
| **Use when** | Sharing task instructions and workflows | Sharing complete agent infrastructure |
| **Activated** | On demand, by the model via `activate_skill` | When the runtime loads |

They are complementary: a package can ship prompts and tactics *and* a
`skills/` folder for agents that prefer skills.

### Writing a package for sharing

**Use a unique, stable package name.** `name` becomes the namespace prefix of
every resource and part of every session's `tactic_path`. Pick something specific
(`acme-finance`, not `finance`) and never rename it after publishing.

```toml
[package]
name = "acme-finance"
version = "1.2.0"
description = "Financial data prompts and analysis tactics"
```

**Qualify references in configs.** Bare names resolve against the *root*
package's namespace, which is not yours when your package is a dependency. Use
package-qualified prompt paths and `base` names in configs you ship:

```yaml
base: acme-finance:common              # not: base: common
agent_configs:
  - name: analyst
    system_prompt_path: acme-finance:analysis/system   # not: analysis/system
```

Inside prompts, bare tool refs and `Function` declarations already resolve
within the prompt's own package. In your README, always show fully qualified
names (`load_prompt("acme-finance:analysis/system")`).

**Declare requirements.** Document the environment variables your tactics and
proxies need, and fail with a clear message when one is missing instead of
letting the error surface mid-call.

**Declare your own dependencies** in `[dependencies]`, and either document them
or bundle them with `export_package(..., bundle_deps=True)`.

**Version releases, not just the manifest.** Consumers usually pin a commit or a
release zip. Tag releases, keep a changelog, and treat renamed resources and
removed tactics as breaking changes.

**Keep prompts and tactics portable.** The most reusable packages work with any
model and agent config: no hardcoded model names, no assumptions about the
consumer's proxies. Ship example configs (`configs/examples/`) for consumers to
copy.

**Provide a minimal example.** A short README with the three lines needed to
load the package and run a tactic.
