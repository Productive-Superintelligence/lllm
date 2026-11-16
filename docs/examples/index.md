# Examples

Complete programs you can copy and adapt. Each one is a single file that
combines several parts of LLLM into a small application, with its output and
a short explanation of why it is built that way. The guide pages teach one
feature at a time; these show how the features fit together.

| Example | What it shows |
| --- | --- |
| [Research brief](research-brief.md) | A planner, parallel researchers with a search tool, and a writer that returns a typed brief with checked citations |
| [Support triage](support-triage.md) | A labeling prompt, tools per route, a refund limit enforced in code, and escalation to a human queue |
| [Writer and reviewer](review-loop.md) | Two agents that iterate until a typed review approves, with a round limit and separate memories |
| [A shared service over HTTP](http-service.md) | A tactic served over HTTP and called by another team's agent as a tool, through `RemoteTactic` |

## Running them

The examples need the `native` extra, and the HTTP example also needs
`server` and `client`:

```bash
pip install "lllm-core[native,server,client]"
export OPENAI_API_KEY=sk-...
python research_brief.py
```

They use `gpt-4o-mini`; change `model_name` in the config to use any other
[model](../models.md). The outputs on these pages come from the docs test
suite, where [`lllm.testing`](../testing.md) answers every model call, so a
real model will word things differently. You can run them the same way,
without a key, by wrapping the program in `lllm.testing.FunctionModel` with
your own replies.

More examples live next to the features they teach: the
[introduction](../index.md) builds a two-agent brief writer,
[Pydantic AI](../pydantic-ai.md) wraps and composes Pydantic AI agents, and
[Runtimes](../runtimes.md) adapts plain functions and existing classes.
