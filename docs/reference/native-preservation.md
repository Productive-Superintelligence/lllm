# Native Preservation Audit

This audit records how the old native LLLM runtime was restored inside the new
protocol-first LLLM design.

The rule for this port is simple: preserve native behavior inside
`lllm.runtimes.native` when practical, but do not let native concepts reshape
the public v2 `Tactic` protocol, FastAPI service contract, Pydantic AI adapter,
or PsiHub package metadata.

## Summary

| Area | Status | v2 Boundary |
| --- | --- | --- |
| Prompt/dialog/message model | Ported | Native-scoped under `lllm.runtimes.native.core`. |
| Native `Agent` loop | Ported | Usable for native workflows; not part of protocol `Tactic`. |
| Native `Runtime` registry | Ported | Native resource registry only; PsiHub owns package manifests. |
| Native `Tactic` | Ported and adapted | Kept as native runtime object; bridged through `NativeTacticAdapter`. |
| Invokers | Ported | Optional dependencies; no eager provider import from `lllm`. |
| Proxies and builtin proxy assets | Ported | Optional/live API behavior stays out of normal tests. |
| Jupyter sandbox | Ported and guarded | Optional `sandbox` extra; lazy import with clear install hint. |
| Computer-use tool helper | Ported | Optional `tools` extra. |
| Shared parser utilities | Already shared | Runtime-agnostic parser module remains outside native. |
| Tactic-as-tool bridge | Adapted | Native can consume protocol tactics as `Function` objects. |

## Ported

The following v1 pieces are restored substantially as native runtime code:

| v1 Module Area | Restored Location | Notes |
| --- | --- | --- |
| Agent/dialog/prompt loop | `lllm.runtimes.native.core.agent`, `dialog`, `prompt` | Preserves named dialogs, fork lineage, parser retries, tool interrupts, and `AgentCallSession`. |
| Runtime/resource registry | `lllm.runtimes.native.core.runtime`, `resource`, `config` | Keeps native prompt/tool/proxy/tactic resource loading separate from PsiHub package ownership. |
| Native tactic machinery | `lllm.runtimes.native.core.tactic`, `tactic_registry`, `tactic_tool`, `native.tactic` | Keeps native call sessions, sub-tactic records, native agent groups, and `@tactictool`. |
| Native invokers | `lllm.runtimes.native.invokers` | LiteLLM remains optional through the `native` extra. |
| Proxy system | `lllm.runtimes.native.proxies` | Proxy managers, prompt templates, tactic endpoints, interpreter tools, and builtin proxy modules are restored. |
| Proxy assets | `lllm.runtimes.native.proxies.builtin` | Google Trends JSON files and Wolfram assumptions Markdown are included as package data. |
| Jupyter sandbox | `lllm.runtimes.native.sandbox` | Restored behind lazy optional imports. |
| Computer-use helper | `lllm.runtimes.native.tools.cua` | Restored as optional native tooling. |
| Utility helpers | `lllm.runtimes.native.utils` | Restored for native parser/proxy/runtime support. |

## Adapted

Some restored pieces were changed so they fit the v2 design:

| Behavior | Adaptation | Reason |
| --- | --- | --- |
| `core.py` collapsed primitive file | Replaced by restored `core/` package. | Python cannot have both `core.py` and `core/`; v1 structure carries the real native runtime. |
| Native imports | Top-level `lllm.runtimes.native` and `core` use lazy exports. | Importing `lllm` should not import LiteLLM, Jupyter, Exa, OpenAI, or other optional integrations. |
| Native-to-v2 boundary | `NativeTacticAdapter` lives in `lllm.runtimes.native.adapter`. | Keeps v2 protocol `Tactic` separate from native `Tactic`. |
| Runtime-owned kwargs and metadata | `NativeTacticAdapter` accepts `run_kwargs` and optional context-metadata forwarding. | Mirrors the useful Pydantic AI surrounding-features bridge without importing Pydantic AI. |
| Protocol tactic as native tool | `tactic_as_function()` wraps a protocol tactic as native `Function`. | Reuses the callable-tool idea without making native depend on Pydantic AI internals. |
| Native model validation | Tool names, public maps, metadata, usage, and mutable inputs use v2 boundary hardening. | Preserves native architecture while keeping public model behavior safe and copy-stable. |
| Optional extras | Restored `native`, `sandbox`, and `tools` extras. | Keeps native features available without making the public protocol heavy. |
| Builtin proxy module paths | Updated builtin proxy loader to `lllm.runtimes.native.proxies.builtin.*`. | v1 top-level proxy paths no longer match the v2 package layout. |
| Native `Tactic` bridge helpers | Added `run`, `arun`, `tactic_name`, and `info()` compatibility aliases. | Allows existing v2 callable-tool and metadata helpers to wrap native tactics cleanly. |

## Cut

Nothing major was intentionally deleted from the v1 native runtime in this
checkpoint. The port favors maximal preservation under the native boundary.

The important cut is architectural rather than file-level:

| Old Assumption | Cut From Public v2 | Reason |
| --- | --- | --- |
| Native runtime as the center of LLLM | Public protocol layer | LLLM now centers on runtime-agnostic protocol `Tactic`. |
| Native resource/package registry owning reusable package metadata | PsiHub package protocol | PsiHub owns `psi.toml`, validation, cards, local hub lifecycle, and ref/config metadata. |
| Provider/tool/sandbox dependencies as core imports | Core `lllm` import path | Optional native integrations should not burden Pydantic AI, plain Python, or service-only users. |

## Deferred

The following areas are restored enough to import and test offline, but need
more focused verification before they can be considered complete:

| Area | Deferred Work |
| --- | --- |
| Live native invoker behavior | Add opt-in LiteLLM/live-provider tests that do not print or store secrets. |
| Builtin proxy integrations | Add pseudo-data tests for request/response/error mapping and opt-in live API examples where credentials exist. |
| Jupyter sandbox behavior | Add timeout/error/artifact/output-capture tests behind sandbox dependencies. |
| Live native-agent service examples | Add an opt-in live provider service example once live invoker behavior is tested. |
| Shared utility extraction | Move parser/proxy/sandbox helpers out of native only when they are clearly useful to Pydantic AI and plain Python tactics without adding complexity. |
| Native streaming/events | Expose only when a clean native implementation exists; otherwise report unsupported capability. |

## Verification

Current checkpoint coverage includes:

- native prompt rendering, extension, metadata isolation, and parser behavior;
- native function schema creation, execution, invalid-name rejection, and schema
  copy isolation;
- native message, function-call, logprob, usage, and dialog boundary validation;
- dialog fork lineage and serialization roundtrip;
- native runtime prompt registry;
- native agent named-dialog management;
- native `Tactic.as_tool()` compatibility with v2 callable-tool wrapping;
- `NativeTacticAdapter` protocol boundary behavior;
- `NativeTacticAdapter` runtime kwargs, context forwarding, metadata injection,
  and native stream bridging;
- offline native service example through FastAPI `/run`;
- protocol tactic to native `Function` bridge.

The checkpoint also passed full LLLM tests, strict docs build, package build,
`twine check`, and installed-wheel native smoke.
