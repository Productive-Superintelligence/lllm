# Native Runtime Guide

This guide has moved into the Runtime section.

- Read the native runtime explanation in [Runtime / Native Runtime](../runtime/native.md).
- Follow the hands-on material in [Native Core](../tutorials/native-core.md).
- See [Native Preservation Audit](../reference/native-preservation.md) for the
  ported, adapted, cut, and deferred v1-native pieces.

The core rule is unchanged: the preserved native runtime owns prompts,
dialogs, agents, invokers, tool interrupts, parser retries, and session traces.
LLLM wraps that machinery as a service-ready `Tactic` when it crosses the v2
boundary.
