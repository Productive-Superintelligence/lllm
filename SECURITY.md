# Security Policy

LLLM wraps user logic and can expose it as HTTP services. Treat tactic inputs,
custom endpoints, generated schemas, and runtime adapters as security-sensitive
surfaces.

## Supported Versions

The active `main` branch receives security fixes.

## Reporting A Vulnerability

Please report suspected vulnerabilities privately to the project maintainers.
Do not open a public issue with exploit details.

Include:

- affected version or commit
- tactic or service adapter involved
- reproduction steps
- expected and actual behavior
- whether the issue involves request validation, streaming, custom endpoints,
  sandbox/proxy utilities, or runtime adapter behavior

## Scope

Security-sensitive areas include:

- tactic input/output validation
- FastAPI service envelopes and custom endpoints
- remote tactic clients and error handling
- proxy and sandbox utilities
- optional runtime adapters that call user-provided code
