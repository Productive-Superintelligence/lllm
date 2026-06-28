# Packages

LLLM does not own `psi.toml`; PsiHub owns package metadata. LLLM contributes
tactic metadata that PsiHub can validate, index, and render in package cards.

```toml
[package]
org = "demo"
name = "echo"
kind = "tactic"
primary = "tactics.echo"

[tactics.echo]
entry = "demo.tactics:EchoTactic"
input = "echo_input"
output = "echo_output"

[services.api]
entry = "demo.app:create_app"
tactic = "echo"
transport = "fastapi"
```

Refs are package resource identifiers:

```text
psi://demo/echo/tactics/echo
psi://demo/echo/services/api
```
