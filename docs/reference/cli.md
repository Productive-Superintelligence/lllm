# CLI

Inspect a tactic:

```bash
lllm inspect examples.echo_service.tactics:build_tactic --json
```

`inspect --json` uses the public tactic metadata export and filters raw
secret-shaped keys from examples and user metadata while preserving refs such as
`api_key_ref`/`apiKeyRef`. Snake_case, kebab-case, and camelCase secret keys
such as `api_key`, `x-api-key`, `apiKey`, `accessToken`, and `clientSecret`
are treated consistently.

Serve a tactic:

```bash
lllm serve examples.echo_service.tactics:build_tactic --port 8000
```

Create a project:

```bash
lllm create plain my-tactic
lllm create pydantic-ai my-agent-service
lllm create native my-native-demo
```

The create command scaffolds a runnable tactic/service project. Package
metadata is prepared later with PsiHub. Project names are normalized to portable
slugs and must not contain percent escapes. Directory paths must be non-empty
and unpadded.
