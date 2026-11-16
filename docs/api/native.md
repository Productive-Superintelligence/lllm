# Native runtime

The objects you build agentic programs from. Import them from `lllm` (`from lllm import Agent, Prompt, Tactic, tool`), except `TacticCallSession`, `AgentCallSession`, `BaseHandler`, `DefaultSimpleHandler`, and `tactic_as_function`, which come from `lllm.runtimes.native`. Guides: [Tactics](../tactics.md), [Agents](../agents.md), [Prompts](../prompts.md), [Tools](../tools.md), [Dialogs](../dialogs.md).

## Tactics

::: lllm.runtimes.native.core.tactic.Tactic
    options:
      heading_level: 3

::: lllm.runtimes.native.core.tactic.TacticCallSession
    options:
      heading_level: 3

## Agents

::: lllm.runtimes.native.core.agent.Agent
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.AgentCallSession
    options:
      heading_level: 3

::: lllm.runtimes.native.core.hints.InferenceHints
    options:
      heading_level: 3

## Prompts

::: lllm.runtimes.native.core.prompt.Prompt
    options:
      heading_level: 3

::: lllm.parsers.BaseParser
    options:
      heading_level: 3

::: lllm.parsers.DefaultTagParser
    options:
      heading_level: 3

::: lllm.parsers.ParseError
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.BaseHandler
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.DefaultSimpleHandler
    options:
      heading_level: 3

## Tools

::: lllm.runtimes.native.core.prompt.tool
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.Function
    options:
      heading_level: 3

::: lllm.runtimes.native.core.const.FunctionCall
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.tactic_as_function
    options:
      heading_level: 3

::: lllm.runtimes.native.core.prompt.MCP
    options:
      heading_level: 3

## Dialogs

::: lllm.runtimes.native.core.dialog.Dialog
    options:
      heading_level: 3

::: lllm.runtimes.native.core.dialog.Message
    options:
      heading_level: 3

## Configs and packages

::: lllm.runtimes.native.core.config.resolve_config
    options:
      heading_level: 3

::: lllm.runtimes.native.core.tactic_registry.build_tactic
    options:
      heading_level: 3

::: lllm.runtimes.native.core.runtime.load_runtime
    options:
      heading_level: 3

::: lllm.runtimes.native.core.resource.load_prompt
    options:
      heading_level: 3

::: lllm.runtimes.native.core.runtime.install_package
    options:
      heading_level: 3
