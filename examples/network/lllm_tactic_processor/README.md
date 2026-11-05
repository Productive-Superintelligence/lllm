# LLLM Tactic Processor

This example shows LLLM network and LLLM composed through their intended boundary:

- LLLM network owns channels, subscriptions, cursors, and event lineage.
- LLLM owns the typed tactic that transforms one event payload.
- The processor reads raw events and writes tactic outputs as analysis events.

Run it from a workspace where `lllm` is importable:

```bash
python examples/lllm_tactic_processor/workflow.py
```
