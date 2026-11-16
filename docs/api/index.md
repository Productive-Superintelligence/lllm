# API reference

Signatures and docstrings, generated from the code. The guide pages explain
how the pieces fit together; come here for exact parameters and defaults.

| Page | Covers |
| --- | --- |
| [Native runtime](native.md) | `Tactic`, `Agent`, `Prompt`, parsers and handlers, `tool`, `Dialog`, `Message`, config and package loaders |
| [Protocol](protocol.md) | The runtime-neutral `Tactic`, `CallContext`, events, `as_tactic`, the Pydantic AI and native adapters, refs, errors |
| [Services](services.md) | `create_tactic_app`, custom endpoints, `RemoteTactic`, request and response models |
| [lllm.hub](hub.md) | Manifests, validation, local hubs, catalogs, cards |
| [Proxies and sandbox](capabilities.md) | `BaseProxy`, proxy tactics, `sandbox_tactic`, `SandboxPolicy` |
| [lllm.testing](testing.md) | `ScriptedModel`, `FunctionModel`, `ModelRequest`, `ToolCall` |
| [CLI](cli.md) | Every `lllm` command and flag |
| [Service API](service-api.md) | The HTTP endpoints of a served tactic, with request and error envelopes |
| [Hub metadata](hub-metadata.md) | The JSON schemas of hub records |

Each page says where to import its names from. Most of what the guides use
comes straight from `lllm`, as in `from lllm import Agent, Prompt, Tactic, tool`.
