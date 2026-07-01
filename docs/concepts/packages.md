# Packages

LLLM does not own `psi.toml`; PsiHub owns package metadata, validation, cards,
agent cards, downloads, and local config templates. LLLM contributes tactic and
service metadata that PsiHub can index.

```toml
[package]
org = "demo"
name = "echo"
kind = "tactic"
primary = "tactics.echo"

[schemas.echo_input]
entry = "demo.schemas:EchoInput"

[schemas.echo_output]
entry = "demo.schemas:EchoOutput"

[tactics.echo]
entry = "demo.tactics:EchoTactic"
input = "echo_input"
output = "echo_output"

[services.api]
entry = "demo.app:create_app"
tactic = "echo"
transport = "fastapi"
```

The package declares importable resources. LLLM can provide the tactic object
and service adapter; PsiHub checks the package shape and renders the package in
a way humans and coding agents can inspect.

## Refs

Package resources get durable refs:

```text
psi://demo/echo/tactics/echo
psi://demo/echo/services/api
```

Refs are identifiers, not launch instructions. A local config file can bind a
tactic ref to a URL, a script can register a local tactic object under the same
ref, and a package card can show the ref before anything is running.

## What To Put In Metadata

Good metadata:

- a concise description of the tactic,
- examples with representative input/output,
- schema refs or generated JSON schemas,
- service refs and custom endpoint descriptions,
- latency, safety, and dependency notes.

Avoid raw secrets, tokens, cookies, passwords, and machine-local credentials.
Use local credential refs such as `api_key_ref` when a package needs to point a
runner at credentials without storing them in the package.

## Typical Flow

1. Build and test the tactic locally.
2. Expose it through a service when it needs an HTTP boundary.
3. Add PsiHub metadata with `psihub init`.
4. Validate the package.
5. Publish or download it through a local hub.

LLLM stays focused on the tactic and service boundary throughout that flow.
