# CLI

`lllm` is the single command. Module command groups (`hub`, `network`,
`shell`) take their own options before the subcommand.

```text
lllm create   NAME [--template T]   new project
lllm init     [PATH]                write lllm.toml
lllm setup    [TARGET]              store the API keys a package declares
lllm inspect  TARGET                tactic metadata, or what serve would launch
lllm serve    TARGET                serve a tactic or a package resource
lllm hub      ...                   validate, publish, read, index packages
lllm network  ...                   SSSN channel stores (experimental)
lllm shell    ...                   AAAX strategies (experimental)
```

`TARGET` is a `module:attribute` entrypoint (imported from the current
directory) or a package path (a folder with `lllm.toml`, or the file itself).

## `lllm create`

```bash
lllm create my-app                         # native project, template "minimal"
lllm create my-app --template pipeline     # pipeline | proxy | research | service
lllm create my-app --template ./my-template  # a folder with lllm-template.toml
lllm create plain my-svc                   # service scaffolds: plain | pydantic-ai | native
```

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
if there is none. See [The manifest](../packages/manifest.md).

## `lllm setup`

```bash
lllm setup ./briefs                       # prompts for missing keys
lllm setup ./briefs --credentials env --env-file .env.local --set OPENAI_API_KEY=sk-...
lllm setup --show-conventions             # common provider key names
```

Reads `[requirements.api_keys]` and stores missing keys in the OS keyring
(`secure` extra) or a local env file. `--credentials auto` uses the keyring
when available.

## `lllm inspect`

```bash
lllm inspect tactics:Shout --json     # name, schemas, capabilities, endpoints
lllm inspect ./briefs                 # resolved resource, port, routes, key status
```

Package options: `--resource section.name`, `--store DIR` (channel
resources), `--env-file`, `--no-keyring`.

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
| `--store` | `<package>/.lllm/network` | Store root for channel resources |
| `--credentials`, `--env-file`, `--no-keyring` | | Key lookup and interactive setup |
| `--skip-key-check` | off | Start without checking declared keys |

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

See [The Hub](../packages/hub.md).

## `lllm network`

```bash
lllm network [--store DIR] COMMAND
```

`--store` defaults to `.lllm/network`. Commands: `channels`,
`create-channel`, `get-channel`, `append`, `query-events`, `get-event`,
`create-subscription`, `get-subscription`, `pull-subscription`,
`write-artifact`, `get-artifact`, `read-artifact`, `put-snapshot`,
`get-snapshot`, `serve`. See [SSSN](../sssn/serving.md).

## `lllm shell`

```bash
lllm shell inspect TARGET [--json]
lllm shell serve TARGET [--host] [--port 8400] [--log-level]
```

`TARGET` is a `strategy.py` file, a folder (with `strategy.py`, `shell.py`, or
`lllm.toml`), or a `module:factory`. See [AAAX](../aaax/index.md).
