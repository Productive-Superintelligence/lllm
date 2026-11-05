# Tutorials

A guided path from a one-line chat to served, packaged, connected agentic
systems. Lessons 1 to 8 teach the native runtime; 9 to 11 take a system
outward.

| # | Lesson | What you will learn |
| --- | --- | --- |
| 1 | [Quick Start](01-quick-start.md) | `Tactic.quick()`, open → receive → respond, local models, `lllm create` |
| 2 | [Agents and Dialogs](02-agents-and-dialogs.md) | The core mental model: `Agent`, `Dialog`, multi-turn chats, forking |
| 3 | [Prompts and Structured Output](03-prompts.md) | Templates, parsers, Pydantic output, exception handlers, prompt inheritance |
| 4 | [Tools, Proxies, and Sandbox](04-tools.md) | `@tool`, the tool loop, MCP, proxies with `run_python` and `CALL_API` |
| 5 | [Native Tactics](05-native-tactics.md) | Multi-agent programs, call sessions, sub-tactics, batch and async |
| 6 | [Packages and the Hub](06-packages-and-hub.md) | `lllm.toml`, discovery, YAML agent configs, validating and publishing |
| 7 | [Logging and Cost Tracking](07-logging.md) | Loggers, call sessions and cost, observers |
| 8 | [Advanced Patterns](08-advanced-patterns.md) | Proxy orchestration, forking, pipelines, streaming, offline testing |
| 9 | [Serving and Composition](09-serving.md) | `lllm serve`, the HTTP API, remote tactics, refs, `lllm.Resolver` |
| 10 | [Pydantic AI Agents](10-pydantic-ai.md) | Pydantic AI agents as tactics, tactics as Pydantic AI tools |
| 11 | [Channels with SSSN](11-channels.md) | Channels, `TacticProcessor`, serving a store, AAAX strategies |

## Prerequisites

- Python 3.10+
- `pip install "lllm-core[native,server]"`
- A model: an API key for a hosted provider (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, ...) or a local OpenAI-compatible server such as vLLM,
  SGLang, or NVIDIA Dynamo. Lesson 1 shows both.

## How to read them

Each lesson builds on the previous one, and every code block is runnable.
Lessons that call a model say so; everything else runs offline. When you want
the full detail behind a lesson, each one links to its reference page.
