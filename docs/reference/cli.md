# CLI

Inspect a tactic:

```bash
lllm inspect examples.echo_service.tactics:build_tactic --json
```

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
