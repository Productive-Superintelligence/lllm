# Sandboxes

Goal: add small application-level guardrails around tactic calls.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Files Used

```text
lllm/
  sandbox.py
tests/
  test_sandbox.py
```

## Policy

```python
from lllm import SandboxPolicy, SandboxedTactic


safe = SandboxedTactic(
    tactic,
    policy=SandboxPolicy(
        timeout_seconds=2.0,
        max_input_bytes=4096,
        max_output_bytes=4096,
        allowed_metadata_keys=("tenant", "trace"),
    ),
)

result = await safe.arun(
    {"text": "hello"},
    context=context,
)
```

`SandboxedTactic` wraps the public `Tactic` boundary, so it can sit around
plain Python, Pydantic AI, native, or remote tactics. It checks request metadata
before the call, input/output payload sizes around the call, and async/service
deadlines with `asyncio.wait_for`.

Metadata allowlist keys are plain string tokens such as `tenant`,
`x-api-key`, or `trace.id`; blank keys, bytes, whitespace, percent escapes, and
path separators are rejected at policy construction time.

This is not OS-level isolation. Use containers, separate processes, or a real
security sandbox for untrusted code. The LLLM sandbox wrapper is for reusable
service guardrails and predictable error types.

## Verify

```bash
python -m pytest tests/test_sandbox.py -q
```

Expected output:

```text
... passed
```

Next, serve the sandboxed tactic so timeout and limit errors travel through the
normal service error envelope.
