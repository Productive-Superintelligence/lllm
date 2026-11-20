# CLI

`lllm` is the single command. Command groups (`hub`, and the plugin commands
`network` and `shell`) take their own options before the subcommand.

```text
lllm create   NAME [--template T]  new project
lllm init     [PATH]               write lllm.toml
lllm setup    [TARGET]             store the API keys a package declares
lllm inspect  TARGET               tactic metadata, or what serve launches
lllm serve    TARGET               serve a tactic, app, or package resource
lllm hub      ...                  validate, publish, read, index packages
lllm network  ...                  SSSN channel stores (pip install sssn)
lllm shell    ...                  AAAX strategies (pip install aaax)
```

`TARGET` is a `module:attribute` entrypoint (imported from the current
directory) or a package path (a folder with `lllm.toml`, or the file itself).

## `lllm create`

```bash
lllm create my-app                           # template "minimal"
lllm create my-app --template research       # or pipeline, proxy, service
lllm create my-svc --template pydantic-ai    # a typed service scaffold
lllm create my-app --template ./my-template  # has lllm-template.toml
```

The templates `minimal`, `pipeline`, `proxy`, `research`, and `service`
create native projects; `plain`, `pydantic-ai`, and `native` create typed
service scaffolds. The older form `lllm create TEMPLATE NAME` also works.

| Option | Default | |
| --- | --- | --- |
| `--template` | `minimal` | Template name or folder |
| `--model` | `gpt-4o-mini` | Model written into `configs/default.yaml` |
| `--directory` | `.` | Parent folder |
| `--force` | off | Overwrite existing files |

Native templates create `lllm.toml`, `prompts/`, `configs/default.yaml`,
`tactics/`, `main.py` (with `create_tactic()`), and an offline smoke test.

## `lllm init`

```bash
lllm init [PATH] [--name NAME] [--org ORG] [--kind KIND] [--force]
```

Writes a minimal `lllm.toml` (with `[card]` and `[docs.readme]`) and a README
if there is none. See [The manifest](../packages.md).

## `lllm setup`

```bash
lllm setup ./briefs                       # prompts for missing keys
lllm setup ./briefs --credentials env --env-file .env.local \
  --set OPENAI_API_KEY=sk-...
lllm setup --show-conventions             # common provider key names
```

Reads `[requirements.api_keys]`, plus the `required_api_keys` metadata of the
resource `lllm serve` would launch (or the one named with `--resource`), and
stores missing keys in the OS keyring (`secure` extra) or a local env file. A
package with nothing to launch, such as a library, is set up from its
`[requirements.api_keys]` alone. `--credentials auto` uses the keyring when
available.

## `lllm inspect`

```bash
lllm inspect tactics:Shout --json  # name, schemas, capabilities, endpoints
lllm inspect app:app               # an app's port and routes
lllm inspect ./briefs              # resource, port, routes, key status
```

Package options: `--resource section.name`, `--env-file`, `--no-keyring`.
For a channel resource, `lllm inspect` lists the routes of an app built
over a throwaway store, so it creates no store, and `--store` has no
effect. Like `lllm serve`, it imports the package's entries and calls its
factories, so inspect only packages you trust.

## `lllm serve`

```bash
lllm serve tactics:Shout --port 8000
lllm serve ./briefs                       # the package's primary resource
lllm serve ./briefs --resource services.api
```

| Option | Default | |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address (host or IP, not a URL) |
| `--port` | declared port or 8000 | |
| `--log-level` | `info` | Uvicorn log level |
| `--resource` | `[package] primary` | Resource to launch |
| `--store` | `<package>/.lllm/network` | Store root for channel resources (needs sssn) |
| `--credentials`, `--env-file`, `--no-keyring` | | Key lookup and interactive setup. Typed keys go to the OS keyring when a usable one is available (`--credentials auto`) and to `--env-file` otherwise; with `--no-keyring` they go to the env file, and `--credentials keyring` is refused when a package's keys are checked |
| `--skip-key-check` | off | Start without checking declared keys |

An entrypoint names a tactic class or instance, a factory, an object with a
`run` method, a handler function, or a FastAPI or Starlette app (or a factory
that returns one), which is served as it is. A resource the manifest does
not declare or that cannot be launched, or a service entry that fails to
import or build, stops with an error that names the cause. See
[Serving](../serving.md).

## `lllm hub`

```bash
lllm hub [--hub DIR] COMMAND
```

`--hub` defaults to `.lllm/hub`.

| Command | |
| --- | --- |
| `init [PATH]` | Same as `lllm init` |
| `validate PATH` | Validate a package |
| `publish PATH --local [--no-validate]` | Publish into the local hub |
| `list` | Published packages (`org/name@version`) |
| `get ID [--dest DIR] [--version V]` | Download a package |
| `card ID`, `agent-card ID` | Render the package card / agent card |
| `config-template ID` | Render a starting `.lllm/config.toml` |
| `index OUTPUT [--generated-at TS]` | Write a static catalog for the hub web app |
| `serve [--host] [--port 8787]` | Serve the hub API |

See [The Hub](../publishing.md).

## Plugin commands

Other packages add commands to `lllm` through the `lllm.commands` entry-point
group, which maps a command name to a module with `main(argv)`:

```toml
[project.entry-points."lllm.commands"]
network = "sssn.cli"
```

`lllm --help` lists the installed ones. A plugin's `main(argv)` runs as it
would under the package's own command: `lllm` imports it from the installed
package and does not put the working directory on `sys.path`, so a local
`sssn.py` or `aaax.py` cannot shadow it. LLLM's companion packages add two:

| Command | Install | Docs |
| --- | --- | --- |
| `lllm network [--store DIR] COMMAND` | `pip install sssn` or `"lllm-core[network]"` | [sssn.one](https://sssn.one/) |
| `lllm shell inspect TARGET`, `lllm shell serve TARGET` | `pip install aaax` or `"lllm-core[shell]"` | [aaax.one](https://aaax.one/) |

Each package also keeps its own command, `sssn` or `aaax`. Without the
package, the `lllm` command says what to install and exits with status 2:

```text
lllm network needs the sssn package: pip install sssn
```
