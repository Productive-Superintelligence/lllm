# The Hub

The hub is how LLLM packages are shared: validated, published, rendered as
cards for people and agents, downloaded, and served. Browse the public hub at
[lllm.one/hub](https://lllm.one/hub/).

The hub works on [lllm.toml](manifest.md) packages and comes in three forms:

| | What it is | Use it for |
| --- | --- | --- |
| Local hub | A folder (`.lllm/hub` by default) managed by `lllm hub` | Publishing and fetching packages on one machine or a shared drive |
| Hub API | `lllm hub serve`, a FastAPI app over a local hub | Sharing a hub on a network |
| Static hub | `lllm hub index`, a catalog plus zips any web server can host | Read-only public listings |

## Validate and publish

```bash
lllm hub validate ./briefs
lllm hub publish ./briefs --local          # copies the package into .lllm/hub
lllm hub list
```

```text
ok
acme/briefs@0.1.0
acme/briefs@0.1.0
```

Publishing validates first and refuses packages with errors. Secrets and
build leftovers (`.env`, `.git`, `__pycache__`, `*.egg-info`, ...) are never
copied. Use `--hub DIR` (before the command) to work with another hub folder:

```bash
lllm hub --hub /shared/lllm-hub publish ./briefs --local
```

## Read a package

```bash
lllm hub card acme/briefs             # Markdown card for people
lllm hub agent-card acme/briefs       # operating notes and refs for coding agents
lllm hub config-template acme/briefs  # a starting .lllm/config.toml
lllm hub get acme/briefs --dest ./vendor
```

Cards are generated from the manifest: description, `[card]` summary and tags,
declared resources with their refs and entrypoints, API-key requirements, and
example calls. The agent card is written for coding agents that will use the
package: it lists every ref and says how to run the package.

## Run a downloaded package

```bash
lllm setup ./vendor/briefs        # store the API keys the package declares
lllm serve ./vendor/briefs        # serve its primary resource over FastAPI
```

`lllm serve` checks `[requirements.api_keys]` before starting, reading keys
from the environment, a local `.env.local`, or the OS keyring (the `secure`
extra). See [Services](../protocol/services.md) for the HTTP API.

Native packages can also be installed as runtime dependencies: a package
archive installed with `lllm.install_package("briefs.zip")` lands in
`~/.lllm/packages/` (or `./lllm_packages/` with `scope="project"`), and
`lllm.load_runtime()` loads it alongside your project. See
[Discovery](discovery.md#sharing-packages).

## Serve a hub

```bash
lllm hub serve --port 8787
```

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Liveness |
| `POST /validate` | Validate a package folder on the server |
| `POST /publish` | Publish a package folder on the server |
| `GET /packages` | List packages |
| `GET /packages/{org}/{name}` | Package record (`?version=` for a specific one) |
| `GET /packages/{org}/{name}/card` | Package card |
| `GET /packages/{org}/{name}/agent-card` | Agent card |
| `GET /packages/{org}/{name}/config-template` | Local config template |
| `GET /packages/{org}/{name}/download` | Package zip |

[lllm.one/hub](https://lllm.one/hub/) is its own website, built on the same
package format: the
[lllm-hub](https://github.com/Productive-Superintelligence/lllm-hub)
repository adds a web app, authenticated uploads, and object storage.

## Publish a static hub

```bash
lllm hub --hub .lllm/hub index ./site/hub
```

```text
site/hub/
  data/catalog.json
  packages/acme-briefs-0.1.0.zip
```

The catalog holds each package's latest version: summary, resources, card,
agent card, config template, README, and download link. To make it browsable,
add the hub web app with `lllm-hub --export-web ./site/hub` (from the
lllm-hub repository); any static host can then serve the folder.

## From Python

```python
from lllm.hub import LocalHub, build_catalog, init_package, validate_package

init_package("briefs", org="acme")          # writes briefs/lllm.toml
report = validate_package("briefs")
assert report.ok, report.issues

hub = LocalHub(".lllm/hub")
record = hub.publish("briefs")
print(record.key, [resource.ref for resource in record.resources])
print(hub.card("acme/briefs"))
build_catalog(hub, "site/hub")
```

`lllm.hub.tactic_resource(tactic)` and
`lllm.network.channel_resource(channel)` export metadata for tactics and
channels you want to describe in a manifest.
