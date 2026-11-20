# Packages

A package is a folder with an `lllm.toml`. The manifest says where the
package's prompts, configs, tactics, and proxies live; LLLM discovers them
and registers each one under the package's name, so code and configs refer
to resources by name instead of by import path. The same file tells
`lllm serve` what to launch and the [hub](publishing.md) what to list.

`lllm create support` writes a working package:

```text
support/
├── lllm.toml              # the manifest
├── prompts/assistant.py   # Prompt(path="system", ...)
├── configs/default.yaml   # the agent: model and system prompt
├── tactics/assistant.py   # AssistantTactic, name = "assistant"
├── main.py                # create_tactic(): load and build the tactic
├── tests/test_smoke.py    # builds the tactic without calling a model
└── README.md, pyproject.toml, .env.example, .gitignore
```

Loading it registers what it declares, and the config builds the tactic:

```python title="load_package.py"
from lllm import build_tactic, create_project, load_runtime, resolve_config

create_project("minimal", "support")  # (1)!
runtime = load_runtime("support/lllm.toml")  # (2)!
print(sorted(runtime.keys()))
"""
[
    'support.configs:default',
    'support.prompts:assistant/system',
    'support.tactics:assistant',
]
"""

assistant = build_tactic(resolve_config("default"))  # (3)!
print(assistant("Write a one-line greeting for new customers."))
#> Welcome aboard! We are glad you are here.
```

1. Writes the `minimal` template into `./support`, as `lllm create support`
   does.
2. `load_runtime` reads the manifest, imports the Python files in the
   folders it lists, and registers each resource under a key of the form
   `package.section:path`. YAML configs are read when first used.
3. `configs/default.yaml` names the tactic (`tactic_type: assistant`) and
   the agent's prompt (`system_prompt_path: assistant/system`). Nothing in
   the code imports either file.

Here is the manifest the template wrote:

```toml title="support/lllm.toml"
[package]
org = "local"
name = "support"
version = "0.1.0"
description = "Support"
primary = "tactics.assistant"

[prompts]
paths = ["prompts"]

[configs]
paths = ["configs"]

[tactics]
paths = ["tactics"]

[tactics.assistant]
entry = "main:create_tactic"
description = "A minimal LLLM project: one agent behind a tactic."

[card]
summary = "A minimal LLLM project: one agent behind a tactic."
tags = ["lllm", "minimal"]

[docs.readme]
path = "README.md"
title = "README"
```

The file has two readers. The native runtime reads `[package]` and the
discovery sections, the tables with `paths`. `lllm serve` and the hub read
the named resources, such as `[tactics.assistant]` with its `entry`, plus
`[card]` and `[docs]`. Neither needs the other's tables, so a package can
start with discovery alone and gain hub metadata later.

## The manifest

`[package]` identifies the package:

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | the folder name | Namespace of every resource; the hub requires it |
| `org` | `"local"` | Owner; `org/name` identifies the package on the hub |
| `version` | `"0.1.0"` | |
| `description` | `""` | One line about the package |
| `primary` | none | The resource `lllm serve` launches, as `section.name` |
| `kind` | `"mixed"` | `tactic`, `channel`, `service`, `app`, `library`, or `mixed` |
| `license`, `authors` | none | |

Discovery sections list folders, relative to the manifest, for the native
runtime to scan. A section without `paths`, or a built-in section with no
table at all, uses the folder with its name if there is one. Files whose
names start with `_`, hidden data files, and `__pycache__` are skipped.

| Section | Registers |
| --- | --- |
| `[prompts]` | Module-level `Prompt` objects |
| `[tools]` | Module-level `@tool` functions; other files as data |
| `[proxies]` | `BaseProxy` subclasses ([Proxies](proxies.md)) |
| `[tactics]` | `Tactic` subclasses that set `name` |
| `[configs]` | YAML files ([Configuration](configuration.md)) |
| `[assets]` | Any files, read when first used |
| `[services]` | Module-level FastAPI or Starlette apps; spec files as data |
| Any other table with `paths` | Files, read when first used: JSON, YAML, and TOML parsed, others as bytes |

The other tables are for `lllm serve` and the hub. Each named resource,
such as `[tactics.NAME]`, gets a [ref](#refs) and accepts
`metadata = {...}`:

| Table | Main fields |
| --- | --- |
| `[tactics.NAME]` | `entry`, `description`, `input` and `output` (schema names), `examples` |
| `[services.NAME]` | `entry` (an app or a function that builds one), `tactic`, `transport` |
| `[schemas.NAME]` | `entry` (a Pydantic model) |
| `[channels.NAME]`, `[snapshots.NAME]` | Channels for [SSSN](https://sssn.one/) |
| `[runs.NAME]` | The services, tactics, and channels that run together |
| `[config]` | `schema` and `defaults` for package settings |
| `[docs.NAME]`, `[examples.NAME]`, `[assets.NAME]` | Files that ship with the package |
| `[requirements.api_keys]` | `NAME = "why it is needed"` ([API keys](#api-keys)) |
| `[card]` | `summary`, `tags`, `safety`, `latency`, `suggested_commands` |

An `entry` is `module:attribute`, imported from the package folder. A
tactic entry is loaded like a tactic target of `lllm serve
module:attribute` ([Serving](serving.md#serve-from-the-command-line)): a
class or a function without parameters is called once, and any other
function answers each input. An app belongs in a `[services.NAME]` entry;
a tactic entry that names one stops `lllm serve` with an error. `lllm hub
validate` checks all of this before you share the package
([The hub](publishing.md#validate-a-package)).

## How resources are named

Every discovered resource has a key, `package.section:path`:

| Section | `path` | Example key |
| --- | --- | --- |
| `prompts` | File path without `.py`, then the prompt's `path` | `support.prompts:assistant/system` |
| `tools` | File path, plus the tool's name if it differs | `support.tools:search` |
| `proxies` | The `@ProxyRegistrator` path, else file path and class | `support.proxies:weather` |
| `tactics` | The tactic's `name` | `support.tactics:assistant` |
| `configs` | File path without `.yaml` | `support.configs:default` |
| `assets` and others | File path with its extension | `support.assets:logo.png` |

Loaders accept the full key, the shorthand `package:path`, or a bare path,
which resolves in the default package, the first one loaded:

```python title="names.py"
from lllm import create_project, load_prompt, load_runtime, load_tactic

create_project("minimal", "support")
load_runtime("support/lllm.toml")

prompt = load_prompt("assistant/system")
print(prompt is load_prompt("support:assistant/system"))
#> True
print(prompt is load_prompt("support.prompts:assistant/system"))
#> True
print(load_tactic("assistant").__name__)
#> AssistantTactic
```

`load_tool`, `load_proxy`, `load_config`, `resolve_config`, and
`load_resource` (any section, by full key) work the same way, and an
unknown name raises a `KeyError` that suggests close matches. In a package
that others load, bare names resolve in *their* default package, so
qualify the names in configs you ship: `system_prompt_path:
support:assistant/system`.

A package can load others first, and a folder can sit under a prefix:

```toml
[dependencies]
packages = ["../shared/style-guide as style"]

[prompts]
paths = ["prompts", "vendor/prompts under vendor"]
```

A dependency loads into its own namespace, reachable by its name and by
the alias (`style:...`), and loads once even if several packages list it.
`under` puts every key from that folder under `vendor/`, so two folders
cannot collide. Tables work too: `{path = "...", alias = "..."}` and
`{path = "...", prefix = "..."}`.

## Start a package

`lllm create NAME` writes a project into `./NAME`. `--template` picks the
starting point, `--model` the model in its configs (default
`gpt-4o-mini`), `--directory` the parent folder, and `--force` overwrites.

| Template | What you get |
| --- | --- |
| `minimal` (default) | One agent behind a tactic |
| `pipeline` | A planner, a writer, and a reviewer in sequence |
| `proxy` | An analyst agent that programs against an API proxy |
| `research` | A research workspace with batch experiments |
| `service` | A tactic plus `service.py`, its FastAPI app |
| `plain`, `pydantic-ai`, `native` | A typed tactic as a service project (`app.py`, `client.py`), without `lllm.toml` |
| A folder | Your own template: a folder with `lllm-template.toml` |

The first five write `lllm.toml`, a `main.py` whose `create_tactic()` is
the package's primary tactic, and a smoke test that builds the tactic
without calling a model. A template folder of your own holds its files
in `template/` and declares `[template] name` and the `[variables]` it uses
in `lllm-template.toml`; `lllm create` fills `{{project_name}}`,
`{{package_name}}`, `{{pyproject_name}}`, `{{project_title}}`, and
`{{model_name}}` in file contents and names.

`lllm init [PATH]` adds a minimal `lllm.toml` (and a README if there is
none) to an existing folder. `--name` defaults to the folder's name,
`--org` to `local`, and `--kind` to `mixed`. An existing manifest is kept
unless you pass `--force`.

## Refs

A ref names a resource independently of where it runs:
`lllm://ORG/PACKAGE/SECTION/NAME`. A package gives each declared resource
a ref from its `org` and `name`, so `[tactics.triage]` in package
`acme/support` is `lllm://acme/support/tactics/triage`. The sections are
`tactics`, `services`, `schemas`, `channels`, `snapshots`, `runs`,
`configs`, `docs`, `examples`, and `assets`.

Refs appear on hub cards, as a tactic's `package_ref` in `info()`, and as
the `tactic_ref` of a request's context. `lllm.TacticRef` parses a tactic
ref and raises `TacticRefError` for anything else. Code that calls tactics
by ref keeps working when a tactic moves from your process to a service,
because local configuration decides what each ref means.

## Local configuration

Where a tactic runs and where a channel keeps its data depend on the
machine, so those bindings live outside the package, in
`.lllm/config.toml`:

```toml title=".lllm/config.toml"
[refs."lllm://acme/support/tactics/triage"]
url = "http://127.0.0.1:8100"          # a served tactic

[refs."lllm://acme/support/channels/tickets"]
store = ".lllm/network"                # a channel store (needs sssn)

[settings]
region = "eu"
```

`lllm.Resolver` reads the file once and turns refs into live objects:

```python title="resolver.py"
from pathlib import Path

from lllm import Resolver, Tactic

TRIAGE = "lllm://acme/support/tactics/triage"
SUMMARIZE = "lllm://acme/support/tactics/summarize"

Path(".lllm").mkdir()
Path(".lllm/config.toml").write_text(
    f"""
[refs."{TRIAGE}"]
url = "http://127.0.0.1:8100"

[settings]
region = "eu"
"""
)


class Summarize(Tactic):
    name = "summarize"

    def call(self, text: str) -> str:
        return text.split(". ")[0] + "."


resolver = Resolver.from_config(".")
resolver.register(SUMMARIZE, Summarize())  # (1)!

triage = resolver.tactic(TRIAGE)  # (2)!
print(type(triage).__name__, triage.url)
#> RemoteTactic http://127.0.0.1:8100/run
print(resolver.run(SUMMARIZE, "The app crashed. Logs attached."))
#> The app crashed.
print(resolver.setting("region"), resolver.setting("timeout", 30))
#> eu 30
```

1. `register()` binds a ref to a tactic in this process and sets its
   `package_ref`. It replaces a binding from the file, which is handy in
   tests.
2. A tactic ref bound to a `url` resolves to a `RemoteTactic` named after
   the ref. Code that calls `resolver.run(TRIAGE, ...)` stays the same
   whether triage runs here or behind that URL.

- `Resolver.from_config(path=".")` takes a project folder (it reads
  `PATH/.lllm/config.toml`) or the file's own path; `Resolver()` starts
  empty.
- A tactic ref binds to `url`: a service's base URL, or `.../tactics/NAME`
  in a multi-tactic app. The remote tactic has no input or output types,
  so it returns JSON as is; `resolver.tactics.bind_url(ref, url, **kwargs)`
  binds one in code and passes the keywords, such as `input_type`, to
  `RemoteTactic`.
- Channel and snapshot refs bind to `store` (a folder in the project) or
  `url` (a store service), and `resolver.store(ref)` opens the store. They
  need [SSSN](https://sssn.one/); tactic refs and settings work without it.
- Other keys of a binding, and its `metadata` table, become the remote
  tactic's `info().metadata`. Raw secrets such as `api_key` are refused;
  name a credential with a reference such as `api_key_ref` instead.
- `resolve(ref)` picks `tactic()` or `store()` by the ref's section,
  `arun()` is the async `run()`, and `refs()` lists every bound ref.

`lllm hub config-template ORG/NAME` writes a starting `.lllm/config.toml`
for a published package ([The hub](publishing.md#cards)).

## Share and install packages

Sharing a package means sharing its folder. `load_runtime()` also loads
every package it finds in two drop-in folders, each into its own
namespace:

| Folder | Scope |
| --- | --- |
| `lllm_packages/NAME/`, next to the project's `lllm.toml` | The project; commit it with the repo |
| `~/.lllm/packages/NAME/` | Every project of the current user |

A package that fails to load is skipped with a warning. `install_package`
puts a package zip into one of these folders and loads it right away:

```python title="install.py"
import shutil

from lllm import create_project, install_package, load_prompt

create_project("minimal", "glossary", directory="downloads")
shutil.make_archive("glossary", "zip", "downloads/glossary")  # (1)!

path = install_package("glossary.zip", scope="project")  # (2)!
print(path.parent.name, path.name)
#> lllm_packages glossary
print(load_prompt("glossary:assistant/system").path)
#> system
```

1. A package zip has `lllm.toml` at its root or in one top-level folder,
   like the downloads from the [hub](publishing.md).
2. `scope="project"` installs into `lllm_packages/`; the default,
   `scope="user"`, installs into `~/.lllm/packages/`.

| Function | Does |
| --- | --- |
| `install_package(zip, *, alias=None, scope="user", load=True)` | Installs and loads a package; `alias` renames it (lowercase letters, numbers, `-`, and `_`, as in `team_glossary`), and an installed name raises `FileExistsError` |
| `export_package(name, path, *, bundle_deps=False)` | Zips a loaded package; `bundle_deps=True` adds its dependencies |
| `list_packages(scope=None)` | Name, version, scope, and path of each package |
| `remove_package(name, *, scope=None)` | Deletes an installed package |

## API keys

A package declares the keys it needs, and `lllm setup` and `lllm serve`
check them:

```toml
[requirements.api_keys]
OPENAI_API_KEY = "Model access for the assistant agent."
```

A resource can add keys in its metadata as `required_api_keys`, either a
table like this one or a list of names. `lllm setup` and `lllm serve` check
the package's keys plus those of the resource being launched.

```bash
lllm setup ./support               # ask for missing keys and store them
lllm setup ./support --credentials env --set OPENAI_API_KEY=sk-...
lllm setup --show-conventions      # common provider key names
```

A key counts as available when it is in the environment, in the env file
(`--env-file`, default `.env.local` in the current folder), or in the OS
keyring. `lllm setup` stores missing keys in the keyring when the `secure`
extra is installed and in the env file otherwise (`--credentials keyring`
or `env` decides), and writes the file readable only by you.
`--set NAME=value` stores a key without asking; without a terminal, a
missing key is an error. `lllm inspect ./support` shows each key as
`ready` or `missing`, never its value. Only the `lllm` command reads the
env file and the keyring; a plain Python program sees the environment.
