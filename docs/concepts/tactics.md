# Tactics

`Tactic` is LLLM's center model. A tactic is a small, typed, service-ready unit
that does one thing well.

The boundary is intentionally narrower than an agent framework. It answers:

- what input shape the tactic accepts,
- what output shape it returns,
- whether it supports streaming,
- how to call it locally or remotely,
- how to describe it to services and packages.

## Design Lineage

The `Tactic` boundary was originally developed from the reusable reasoning units
needed by Analytica. Analytica decomposes complex societal, economic, political,
and scientific questions into grounded subpropositions, runs specialized agents
or tools over those subquestions, and synthesizes the results into a more stable
analysis.

LLLM generalizes that pattern. A tactic is not tied to soft propositional
reasoning, forecasting, or any one agent architecture. It keeps the useful
boundary: a typed unit of work that can be validated, streamed, served,
packaged, composed, audited, and reused by other systems.

## Applied Systems

LLLM has also been applied in autonomous software-development systems such as
Apeiron. Apeiron uses agentic components for full-lifecycle application
synthesis, including demand modeling, computer-use agent evaluation, activity
tracing, and locality-controlled iteration. That is the other side of the
tactic design: the same small callable boundary can serve analysis agents,
software-building agents, evaluators, tracers, tools, and deployment services
without forcing them into one runtime.

```python
from pydantic import BaseModel
from lllm import Tactic


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        return EchoOutput(text=input_value.text.upper())
```

Input may arrive as a model instance or a JSON-compatible mapping. The tactic
normalizes the value before `_run()` sees it, so service clients, test code,
and in-process callers can share the same behavior.

## Call Paths

| Method | Use |
| --- | --- |
| `run(input_value, context=None)` | Synchronous local call. |
| `arun(input_value, context=None)` | Async local call. |
| `stream(...)` / `astream(...)` | Data chunks when the tactic supports streaming. |
| `events(...)` / `aevents(...)` | Full `TacticEvent` envelopes for status/error streams. |
| `info()` | `TacticInfo` metadata for services, packages, and agents. |

`CallContext` carries request ids, trace ids, caller identity, and metadata. It
is optional for small scripts and useful for services that need auditability or
cross-service correlation.

## Metadata

`TacticInfo` is the public description of the tactic:

- name and description,
- input and output JSON schemas,
- examples and metadata,
- package and service refs,
- capability flags such as streaming support.

This is what `/info` returns from a service and what PsiHub can render into
package cards. Keep `info()` data descriptive and portable; do not put raw
secrets or environment-specific credentials in metadata.

## Runtime-Agnostic By Design

A tactic can hide:

- a Pydantic AI agent,
- a native prompt/dialog workflow,
- a plain Python function,
- a guardrailed proxy,
- a remote HTTP service.

The caller still sees the same typed boundary. That is why LLLM composes well
with PsiHub and SSSN: the package or channel layer can point at a tactic
without inheriting the implementation runtime.

## Reference

- Cheng, Junyan, Kyle Richardson, and Peter Chin. "Analytica: Soft Propositional
  Reasoning for Robust and Scalable LLM-Driven Analysis." *The Fourteenth
  International Conference on Learning Representations (ICLR)*, 2026.
- Cheng, Junyan, Ankit Srivastava, Jessie Zeng, Milenko Drinic, and Jack W.
  Stokes. "Apeiron: A Scalable LLM-agentic Framework for Autonomous
  Full-lifecycle Demand-optimized Application Synthesis." *Findings of the
  Association for Computational Linguistics: ACL 2026*, 2026, pp. 3868-3899.
