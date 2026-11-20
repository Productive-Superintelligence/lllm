# Protocol

The runtime-neutral boundary every tactic shares, whatever runtime it uses. Import these from `lllm`, except the base class `Tactic`: it is `lllm.protocol.Tactic`, while `lllm.Tactic` is the native tactic, which subclasses it. Guides: [Typed I/O](../typed-io.md), [Other runtimes](../runtimes.md), [Pydantic AI](../pydantic-ai.md).

## Tactic

::: lllm.protocol.tactic.Tactic
    options:
      heading_level: 3

::: lllm.protocol.context.CallContext
    options:
      heading_level: 3

::: lllm.protocol.tactic.CallResult
    options:
      heading_level: 3

::: lllm.protocol.events.TacticEvent
    options:
      heading_level: 3

::: lllm.protocol.tactic.TacticInfo
    options:
      heading_level: 3

## Other runtimes

::: lllm.runtimes.python.as_tactic
    options:
      heading_level: 3

::: lllm.runtimes.native.adapter.NativeTacticAdapter
    options:
      heading_level: 3

::: lllm.runtimes.pydantic_ai.PydanticAITactic
    options:
      heading_level: 3

::: lllm.runtimes.pydantic_ai.tactic_as_tool
    options:
      heading_level: 3

## Refs

::: lllm.protocol.refs.TacticRef
    options:
      heading_level: 3

::: lllm.resolver.Resolver
    options:
      heading_level: 3

## Errors

::: lllm.protocol.errors.LLLMError
    options:
      heading_level: 3

::: lllm.protocol.errors.TacticError
    options:
      heading_level: 3

::: lllm.protocol.errors.TacticUnsupportedError
    options:
      heading_level: 3

::: lllm.protocol.errors.TacticLoadError
    options:
      heading_level: 3

::: lllm.protocol.errors.TacticRefError
    options:
      heading_level: 3

::: lllm.protocol.errors.SchemaError
    options:
      heading_level: 3

::: lllm.protocol.errors.OutputSchemaError
    options:
      heading_level: 3

::: lllm.protocol.errors.ProtocolError
    options:
      heading_level: 3

::: lllm.protocol.errors.TacticServiceError
    options:
      heading_level: 3
