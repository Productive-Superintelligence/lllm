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
| `name` | folder name (the hub requires it) | Package name; the namespace prefix for discovered resources |
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
| `[configs]` | YAML configs (`.yaml`/`.yml`: agent configs, with `base` inheritance) |
| `[services]` (`paths`) | ASGI apps in Python modules; spec files as data |
| `[assets]` (`paths`) | any files, loaded lazily |
| any other table | a custom section: files and Python resources |

A section without `paths` falls back to a folder with the section's name, if
one exists. The native runtime does this for all seven built-in sections, even
with no table. Validation checks that the following stay inside the package:

- the `paths` or fallback folder of `[prompts]`, `[tools]`, `[proxies]`,
  `[configs]`, and custom sections;
- the `paths` of `[tactics]`, `[services]`, and `[assets]`.

It does not check the fallback folder of `[tactics]`, `[services]`, or
`[assets]` without `paths`, or of a built-in section with no table. The native
runtime treats the hub's own tables (`[schemas]`, `[docs]`, `[card]`, ...) as
custom sections only when they set `paths`. The hub cannot load most of them
with `paths`. It does load `[config]` and `[card]` with `paths`, but
validation does not check those paths. Give custom sections their own names.

Resources are keyed `package.section:path`, for example
`briefs.prompts:writer/system`.
`[dependencies] packages = ["./pkg as alias"]` loads other packages first,
under an optional alias.

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
| `[runs.<name>]` | `services`, `tactics`, `channels`, `snapshots` that make up one run; `lllm serve` launches a run that lists exactly one service |
| `[config]` | `schema` and `defaults` for package settings |
| `[docs.<name>]` | `path`, `title` |
| `[examples.<name>]` | `path` or `command` |
| `[assets.<name>]` | `path`, `media_type` |
| `[requirements.api_keys]` | `NAME = "why it is needed"`, checked by `lllm setup` and `lllm serve` |
| `[card]` | `summary`, `tags`, `safety`, `latency`, `suggested_commands` |

The named resource tables, `[config]`, and `[card]` also accept
`metadata = {...}`. The package ships `lllm.toml` as written, so validation
rejects raw secret-shaped keys (`api_key`, `token`, `clientSecret`, ...) in
`metadata`, nested maps included, and in extra keys written directly in those
tables. References such as `api_key_ref` are allowed. Validation does not look
inside maps under JSON-schema keys (`schema`, `input_schema`, `output_schema`,
or forms such as `inputSchema`), since a schema may name a field `api_key`. It
does not check `[config] defaults` or tactic `examples` either. Keep secret
values out of all of these: the package zip keeps them, and the hub API shows
schema maps unchanged.
[Hub Metadata](../reference/hub-metadata.md#rules-for-published-metadata)
lists the key shapes and what cards, the hub API, and the catalog drop.

### Entrypoints

Entrypoints are `module:attribute`, imported relative to the package folder.
A tactic entry may name a `Tactic` class or instance, a factory, or a
handler. A factory is called once, with no arguments, when the package loads;
a handler is wrapped with `as_tactic` and called once per input. `lllm serve`
and AAAX package tactics tell them apart by one rule:

- Any other class is a factory when it can be constructed without arguments.
- Any other callable is a factory when it takes no parameters, like
  `main:create_tactic` in generated projects, or when it can be called
  without arguments and is annotated to return a `Tactic` or a FastAPI or
  Starlette app.
- Everything else is a handler: `def greet(name="world")` answers each input
  instead of being called once at load.

A return annotation that cannot be evaluated at runtime, such as a class
imported only under `TYPE_CHECKING`, counts when its class name ends in
`Tactic` or is `FastAPI` or `Starlette`. A `[services.<name>]` entry has no
handler form: it names an app, or a callable that can be called without
arguments and builds a FastAPI or Starlette app, a tactic, or a data payload.
Anything else is a launch error.

Serving keeps a package's modules imported, and packages served in one
process share `sys.modules`. A module that a tactic imports when it is called
needs a top-level name no other served package uses.

## Validation

```bash
lllm hub validate .
```

Validation imports declared entrypoints without calling them, checks refs and
schema links, checks that declared files and the discovery folders listed in
[Native discovery](#native-discovery) stay inside the package, and rejects
secret-shaped metadata keys. It reports errors (the package cannot be
published) and warnings. The main codes:

| Code | Level | Meaning |
| --- | --- | --- |
| `manifest_load_failed` | error | `lllm.toml` does not parse or fit the manifest, for example a `primary` that is not `section.name`, or `paths` in `[schemas]` |
| `entrypoint_import_failed` | error | A declared `entry` could not be imported |
| `primary_missing` | error | `primary` names a resource the manifest does not declare |
| `schema_ref_missing` | error | A tactic's `input`/`output` names no schema |
| `metadata_secret_key` | error | `metadata` or an extra table key holds a raw secret-shaped key outside a JSON schema |
| `discovery_path_absolute`, `discovery_path_outside_package` | error | A checked discovery path or fallback folder is absolute, escapes the package, or goes through a symlink |
| `discovery_path_missing` | warning | A checked discovery path does not exist |
| `discovery_section_empty` | warning | A checked section has no `paths` and no folder: often a typo such as `[tactic.x]` |
| `dependency_manifest_missing` | warning | A dependency folder has no `lllm.toml` |
| `card_metadata_missing`, `card_summary_missing`, `readme_missing`, `readme_doc_missing` | warning | The hub page will be sparse |

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
