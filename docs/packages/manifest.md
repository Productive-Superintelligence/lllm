# The lllm.toml Manifest

A package is a folder with an `lllm.toml`. The same file has two readers:

- the **native runtime** reads it to discover resources: prompts, tools,
  proxies, tactics, YAML configs, and package dependencies
  (see [Discovery](discovery.md));
- the **hub** reads it to identify, validate, render, publish, and serve the
  package (see [The Hub](hub.md)).

Neither reader needs the other's sections, so a package can start as a
handful of discovery paths and grow hub metadata later. `lllm create` writes
a manifest that already serves both, and `lllm init` writes a minimal one.

## A complete example

```toml
[package]
org = "acme"
name = "briefs"
version = "0.1.0"
description = "Writes and reviews technical briefs."
primary = "tactics.brief"

# --- native discovery -------------------------------------------------
[prompts]
paths = ["prompts"]

[configs]
paths = ["configs"]

[tactics]
paths = ["tactics"]

[dependencies]
packages = ["./vendor/style-guide as style"]

# --- declared resources (hub) ------------------------------------------
[schemas.brief_input]
entry = "briefs.schemas:BriefInput"

[schemas.brief_output]
entry = "briefs.schemas:BriefOutput"

[tactics.brief]
entry = "main:create_tactic"
input = "brief_input"
output = "brief_output"
description = "Plan, draft, and review a brief."

[services.api]
tactic = "brief"
transport = "fastapi"

[requirements.api_keys]
OPENAI_API_KEY = "Model access for the writer and reviewer agents."

[card]
summary = "Technical briefs with a writer and a reviewer agent."
tags = ["writing", "multi-agent"]

[docs.readme]
path = "README.md"
title = "README"
```

`[tactics]` holds both kinds of entry at once: `paths` tells the native runtime
which folder to scan, and `[tactics.brief]` declares one tactic by name for the
hub. The same split works in `[services]` and `[assets]`.

## `[package]`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | folder name | Package name; the namespace prefix for discovered resources |
| `org` | `"local"` | Owner; the identifier is `org/name` |
| `version` | `"0.1.0"` | Package version |
| `description` | `""` | One line about the package |
| `primary` | none | The resource `lllm serve <package>` launches, e.g. `tactics.brief` |
| `kind` | `"mixed"` | `tactic`, `channel`, `service`, `app`, `library`, or `mixed` |
| `license`, `authors` | none | Optional metadata |
| `manifest_version` | `"0.1"` | Manifest format version |

## Native discovery

Discovery sections name folders to scan. Each entry is a path, optionally with
a virtual prefix:

```toml
[prompts]
paths = ["./prompts", "./shared/prompts under shared"]  # or {path = ..., prefix = ...}
```

| Section | Discovers |
| --- | --- |
| `[prompts]` | `Prompt` objects in Python modules |
| `[tools]` | `@tool` functions (plus metadata files) |
| `[proxies]` | `BaseProxy` subclasses |
| `[tactics]` (`paths`) | `Tactic` subclasses with a `name` |
| `[configs]` | YAML/JSON/TOML configs (agent configs, with `base` inheritance) |
| `[services]` (`paths`) | ASGI apps in Python modules; spec files as data |
| `[assets]` (`paths`) | any files, loaded lazily |
| any other table | a custom section: files and Python resources |

A section without `paths` falls back to a folder with the section's name, if
one exists. Resources are keyed `package.section:path`, for example
`briefs.prompts:writer/system`. `[dependencies] packages = ["./pkg as alias"]`
loads other packages first, under an optional alias.

[Discovery](discovery.md) covers keys, aliases, shared packages, and
installing package archives.

## Declared resources

Declared resources are named tables the hub indexes. Each gets a ref,
`lllm://org/package/<section>/<name>`.

| Table | Key fields |
| --- | --- |
| `[schemas.<name>]` | `entry` (a `module:Model`), `description` |
| `[tactics.<name>]` | `entry`, `input`, `output` (schema names), `runtime`, `description`, `examples` |
| `[services.<name>]` | `tactic` or `tactics`, `entry`, `transport`, `subscribes`, `publishes` |
| `[channels.<name>]` | `schema`, `form` (`log`, ...), `description` |
| `[snapshots.<name>]` | `schema`, `channel` |
| `[runs.<name>]` | `services`, `tactics`, `channels`, `snapshots` to start together |
| `[config]` | `schema` and `defaults` for package settings |
| `[docs.<name>]` | `path`, `title` |
| `[examples.<name>]` | `path` or `command` |
| `[assets.<name>]` | `path`, `media_type` |
| `[requirements.api_keys]` | `NAME = "why it is needed"`, checked by `lllm setup` and `lllm serve` |
| `[card]` | `summary`, `tags`, `safety`, `latency`, `suggested_commands` |

Every table also accepts `metadata = {...}`; keys that look like secrets are
rejected.

Entrypoints are `module:attribute`, imported relative to the package folder.
A tactic entry may name a `Tactic` class, an instance, a zero-argument factory
(as `main:create_tactic` in generated projects), or a plain callable.

## Validation

```bash
lllm hub validate .
```

Validation imports declared entrypoints, checks refs and schema links, checks
that declared files stay inside the package, and checks discovery paths. It
reports errors (the package cannot be published) and warnings:

| Code | Level | Meaning |
| --- | --- | --- |
| `entrypoint_import_failed` | error | A declared `entry` could not be imported |
| `primary_missing`, `primary_invalid` | error | `primary` points at nothing |
| `schema_ref_missing` | error | A tactic's `input`/`output` names no schema |
| `discovery_path_outside_package` | error | A discovery path escapes the package |
| `discovery_path_missing` | warning | A discovery path does not exist |
| `discovery_section_empty` | warning | A section has no `paths` and no folder: often a typo such as `[tactic.x]` |
| `dependency_manifest_missing` | warning | A dependency folder has no `lllm.toml` |
| `card_metadata_missing`, `readme_doc_missing` | warning | The hub page will be sparse |

## Local configuration

Bindings that depend on where a package runs (service URLs, channel stores,
settings) live outside the package, in `.lllm/config.toml`:

```toml
[refs."lllm://acme/briefs/tactics/brief"]
url = "http://127.0.0.1:8000"

[refs."lllm://acme/briefs/channels/events"]
store = ".lllm/network"
```

`lllm hub config-template acme/briefs` renders a starting point, and
[`lllm.Resolver`](../protocol/refs.md) reads it.
