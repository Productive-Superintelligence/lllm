# The hub

The [LLLM Hub](https://lllm.one/hub/) lists [packages](packages.md):
tactics, services, and channels, with the prompts and configs behind them.
Every package page is generated from the package's `lllm.toml`: a card
with its refs and the API keys it needs, notes for coding agents, a
starting local config, the README, and a download. `lllm.hub` runs the
same machinery on your machine, so you can validate a package, publish it
to a local hub, read its cards, and build a static catalog before anyone
else sees it.

```python title="publish.py"
from pathlib import Path

from lllm.hub import LocalHub, validate_package

package = Path("triage")
package.mkdir()
(package / "lllm.toml").write_text(
    """
[package]
org = "acme"
name = "triage"
version = "0.1.0"
description = "Route support tickets to the right queue."
primary = "tactics.triage"

[tactics.triage]
entry = "triage_app:Triage"
description = "Route a support ticket to a queue."

[requirements.api_keys]
OPENAI_API_KEY = "Model access for the router agent."

[card]
summary = "Routes tickets to billing, bugs, or on-call."
tags = ["support", "routing"]
"""
)
(package / "triage_app.py").write_text(
    """
from lllm import Tactic


class Triage(Tactic):
    name = "triage"

    def call(self, text: str) -> str:
        return "oncall" if "outage" in text.lower() else "support"
"""
)

report = validate_package("triage")  # (1)!
print(report.ok, [issue.code for issue in report.issues])
#> True ['readme_missing']

hub = LocalHub(".lllm/hub")  # (2)!
record = hub.publish("triage")
print(record.key)
#> acme/triage@0.1.0
print([resource.ref for resource in record.resources])
#> ['lllm://acme/triage/tactics/triage']
```

1. Validation imports each declared `entry` without calling it, and checks
   refs, paths, and metadata. Errors block publishing; warnings, like this
   missing README, do not.
2. A local hub is a folder, `.lllm/hub` by default. `publish()` validates
   the package again, copies it to `packages/acme/triage/0.1.0/`, and adds
   it to the index.

The same from the command line:

```bash
lllm hub validate ./triage
lllm hub publish ./triage --local
lllm hub list
```

```text
warning: readme_missing: Package should include README.md.
ok
acme/triage@0.1.0
acme/triage@0.1.0
```

## Use a package from the hub

Each package on [lllm.one/hub](https://lllm.one/hub/) can be downloaded as
a zip with `lllm.toml` at its root. Unpack it, store the keys it declares,
and serve it:

```bash
curl -L -o analyst-tactics.zip \
  https://lllm.one/hub/api/packages/society/analyst-tactics/download
unzip analyst-tactics.zip -d analyst-tactics
lllm setup analyst-tactics     # store the API keys it declares
lllm serve analyst-tactics     # serve its primary resource
```

To use its prompts, configs, and tactics in your own program instead,
install the zip with `install_package("analyst-tactics.zip",
scope="project")` ([Packages](packages.md#share-and-install-packages)).
Uploading to lllm.one needs the hub's admin token, so its maintainers
publish packages there. Everything else on this page you can run yourself.

## Validate a package

`lllm hub validate PATH` prints each issue and then `ok` or `failed`, and
exits with status 1 when there are errors. `validate_package(path)` returns
the same report, with `ok` and a tuple of issues that each have a `level`,
a `code`, a `message`, and the `resource` concerned. The main codes:

| Code | Level | Meaning |
| --- | --- | --- |
| `manifest_load_failed` | error | `lllm.toml` does not parse or does not fit the format |
| `entrypoint_import_failed` | error | An `entry` cannot be imported |
| `primary_missing` | error | `primary` names a resource that is not declared |
| `schema_ref_missing` | error | A tactic's `input` or `output` names no declared schema |
| `service_tactic_missing` | error | A service's `tactic` is not declared |
| `metadata_secret_key` | error | Metadata holds a raw secret such as `api_key` ([below](#how-metadata-is-checked)) |
| `discovery_path_outside_package` | error | A discovery folder leaves the package or goes through a symlink |
| `discovery_path_missing`, `discovery_section_empty` | warning | A discovery folder does not exist, often a typo |
| `card_metadata_missing`, `readme_missing`, `readme_doc_missing` | warning | The package page would be sparse |

## Publish to a local hub

`lllm hub publish PATH --local` validates the package and copies it into
the local hub; `--no-validate` skips validation. The CLI publishes only to
local hubs, so `--local` is required. `--hub DIR`, given before the
command, picks another hub folder, such as a shared drive:

```bash
lllm hub --hub /shared/lllm-hub publish ./triage --local
lllm hub --hub /shared/lllm-hub get acme/triage --dest ./vendor
```

Each version is kept in its own folder, and `get ID [--dest DIR]
[--version V]` copies the latest version, or the one you name, out of the
hub. Publishing never copies secrets or build leftovers: `.env` files
(except templates such as `.env.example`), `.git`, `.venv`, `.lllm`,
`__pycache__`, `build`, `dist`, `*.egg-info`, SSH keys, `.netrc`,
`.pypirc`, and symlinks stay behind.

## Cards

The hub renders three documents from a published package. The card is for
people:

````text
$ lllm hub card acme/triage
# acme/triage

Routes tickets to billing, bugs, or on-call.

- Version: `0.1.0`
- Kind: `mixed`
- Validation: `ok`

- Tags: `support`, `routing`

## Required API Keys

- `OPENAI_API_KEY`: Model access for the router agent.

## Resources

- `lllm://acme/triage/tactics/triage` (tactic)
  - Route a support ticket to a queue.
  - Metadata: `runtime="python"`

## Local Config Template

```toml
[refs."lllm://acme/triage/tactics/triage"]
url = "http://127.0.0.1:8000/tactics/triage"

[refs."lllm://acme/triage/tactics/triage".metadata]
runtime = "python"
```
````

`lllm hub agent-card acme/triage` writes the same facts for a coding agent
that will use the package: how to set it up and serve it, every ref with
its entry, and the config template. `lllm hub config-template acme/triage`
prints only the template, a starting `.lllm/config.toml` that binds each
tactic and service ref to a URL on `127.0.0.1`, at the port `lllm serve`
uses for it ([Serving a package](serving.md#serving-a-package)); edit the
URLs to match where you serve them
([Local configuration](packages.md#local-configuration)). In Python, they
are `hub.card(id)`, `hub.agent_card(id)`, and `hub.config_template(id)`,
each with an optional `version=`.

Cards show what the manifest says, so write it for readers: a
`description`, a `[card]` with a `summary` and `tags`, a description for
each resource, and `[card] safety`, `latency`, and `suggested_commands`
where they help.

## Serve a hub

`lllm hub serve` puts a local hub behind an HTTP API, on port 8787 by
default (`--host`, `--port`, `--log-level`):

| Route | Returns |
| --- | --- |
| `GET /health` | `{"ok": true}` |
| `GET /packages` | Every package record |
| `GET /packages/{org}/{name}` | One record; `?version=` picks a version |
| `GET /packages/{org}/{name}/card`, `/agent-card`, `/config-template` | The rendered documents |
| `GET /packages/{org}/{name}/download` | The package as a zip |
| `POST /validate`, `POST /publish` | Validate or publish a package folder on the server, given `{"path": ...}` |

`lllm.hub.create_app(hub=None, hub_root=".lllm/hub")` builds the same
FastAPI app, to mount or wrap in your own. It has no authentication of its
own, and `/publish` reads folders on the server, so keep it on a trusted
network or put it behind your own auth.

## Static catalogs

`lllm hub index` writes a static, read-only copy of a local hub: a JSON
catalog and one zip per package, which any web server can host.

```bash
lllm hub index site/hub
```

```text
indexed 1 package(s) into site/hub

site/hub/
├── data/catalog.json
└── packages/acme/triage/0.1.0.zip
```

The catalog holds the latest version of each package: its summary, tags,
resources and refs, validation report, card, agent card, config template,
README, and download link. Each zip holds the package files at its root,
so `unzip 0.1.0.zip -d triage` leaves `lllm.toml` in `triage/`.
`--generated-at` fixes the catalog's timestamp for reproducible builds,
and `build_catalog(hub, output_dir, generated_at=None)` does the same from
Python.

## How metadata is checked

A package ships its `lllm.toml` as written, so validation refuses raw
secrets in metadata. A reference to a secret is fine:

```python title="secrets.py"
from pathlib import Path

from lllm.hub import validate_package

Path("leaky").mkdir()
Path("leaky/lllm.toml").write_text(
    """
[package]
org = "acme"
name = "leaky"

[card]
summary = "Shows what validation refuses."
metadata = {api_key = "sk-live-123", api_key_ref = "openai.default"}
"""
)
report = validate_package("leaky")
print(report.ok)
#> False
for issue in report.issues:
    if issue.level == "error":
        print(issue.code, issue.resource)
        #> metadata_secret_key card
```

The message says what to do: `Metadata must not include raw secret key
'api_key': lllm.toml is published as written. Use a reference such as
'api_key_ref' instead.`

Validation looks at the `metadata` and extra keys of every resource table,
`[config]`, and `[card]`, nested maps included. It does not look inside
JSON schemas (where a field may be called `api_key`), `[config] defaults`,
or tactic `examples`, so keep secret values out of those yourself. Cards,
the hub API, and catalogs also drop secret-shaped keys from what they
show. [Hub metadata](api/hub-metadata.md) lists the exact key patterns and
the `tactic_resource()` helper that exports a tactic's metadata for a
manifest.
