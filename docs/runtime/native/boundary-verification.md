# Boundary And Verification

The native runtime is intentionally rich, but its concepts stay inside the
native runtime unless they cross through a v2 tactic adapter.

## V2 Boundary

| Old Assumption | v2 Treatment |
| --- | --- |
| Native runtime is the center of LLLM | Native is one runtime behind the protocol `Tactic` boundary. |
| Native resources define public package metadata | PsiHub owns `psi.toml`; native package discovery still reads `lllm.toml`. |
| Provider, sandbox, and proxy deps are core deps | They are lazy optional integrations. |
| Live builtin proxies are normal tests | Builtins are restored, but live API behavior stays opt-in and credential-gated. |
| Native agent sessions are protocol fields | Sessions remain native diagnostics; the public protocol returns typed tactic results. |
| Streaming/events are always available | Native reports stream/event support only when the tactic implementation provides it. |

This lets the v1 runtime stay preserved without forcing Pydantic AI, plain
Python, service-only, or package-index users to inherit native-specific
concepts.

## Examples

- `examples/native_dialog/demo.py` shows prompt/dialog lineage.
- `examples/native_service/` serves an offline native prompt/dialog workflow
  through the v2 FastAPI API.
- `docs/tutorials/native-core.md` walks through Prompt, Tools, Parser, Dialog,
  Runtime, Agent, and Native Tactic usage.
- `docs/reference/native-preservation.md` records what was ported, adapted,
  cut from public v2, or deferred for live verification.

```bash
uvicorn app:app --app-dir examples/native_service --reload
```

## Install

Install provider support only when you need live model calls:

```bash
python -m pip install -e ".[native]"
```

## Verify

Useful checks while editing native docs or runtime code:

```bash
uv run --extra docs mkdocs build --strict --site-dir /tmp/lllm-native-site
uv run --extra dev python -m pytest tests/test_docs_rendering.py -q
```
