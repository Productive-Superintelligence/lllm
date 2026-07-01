# Pydantic AI Guide

This guide has moved into the Runtime section.

- Read the runtime explanation in [Runtime / Pydantic AI](../runtime/pydantic-ai.md).
- Follow the step-by-step build in [Pydantic AI Runtime Tutorial](../tutorials/pydantic-ai-runtime.md).

The core rule is unchanged: Pydantic AI owns agent execution, tools, provider
settings, streaming, tracing, evals, graphs, and durable runtime state. LLLM
wraps the agent as a reusable `Tactic` for services and package metadata.
