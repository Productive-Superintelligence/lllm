# 8. Advanced Patterns

**Goal:** combine everything so far into patterns for production-grade agentic
systems: multi-proxy data analysis, hypothesis exploration with dialog forks,
typed multi-stage pipelines, high-throughput batches, streaming, custom repair
strategies, image input, and offline tests.

Each pattern leans on a native capability from the earlier lessons: forkable
dialogs, prompts as per-turn contracts with parsers and handlers, proxies, the
invoker abstraction, and per-call sessions. Pydantic shows up wherever stages
hand data to each other: typed tactic inputs and outputs, and models validated
from parsed replies.

## Prerequisites

Finish [Lessons 2 to 7](02-agents-and-dialogs.md). The proxy basics
(`BaseProxy`, `exec_env: interpreter`, `run_python`, `CALL_API`) are in
[Lesson 4](04-tools.md), and the `JsonlRunLog` observer used in Pattern 4 is
defined in [Lesson 7](07-logging.md), Step 5.

---

## Pattern 1: Multi-proxy data analysis

Give one agent several data sources through a shared interpreter, so it can
fetch and join data across APIs in Python.

LLLM ships proxies for research data. Importing them registers them in the
default runtime:

```python
from lllm.runtimes.native.proxies import load_builtin_proxies

loaded, errors = load_builtin_proxies()
print(errors)   # proxies whose optional packages are missing, e.g. exa needs `exa_py`
```

Each proxy reads its API key from the environment, for example `FMP_API_KEY`,
`FRED_API_KEY`, and `SEARCH_API_KEY` (Google Trends), adds it to its requests,
and caches responses on disk by default. Then activate several of them on one
agent:

```yaml
global:
  model_name: gpt-4o

agent_configs:
  - name: research_analyst
    system_prompt: >
      You are a financial research analyst with company data, macroeconomic
      series, and search-interest data. Combine them in your analysis.
    max_interrupt_steps: 12
    proxy:
      activate_proxies: [fmp, fred, gt]
      exec_env: interpreter
      max_output_chars: 8000
      timeout: 90.0
```

The built-in proxy instructions include the full API directory of every
activated proxy; for these three it runs to tens of thousands of characters.
Activate only the proxies an agent needs, or set `prompt_template`
([Lesson 4](04-tools.md), Step 14) so the agent reads documentation through
`query_api_doc` instead.

The tactic looks like the one in Lesson 4, Step 14: open a dialog, receive the
task, and respond; the proxy tools are offered on every turn. A typical session
interleaves documentation lookups with Python, calling endpoints by the paths
the documentation lists. This is an illustration of the agent's turns, not a
script to run:

```python
# Turn 1 (tool call): read the documentation of a data source
query_api_doc("fred")

# Turn 2 (run_python): pull data from several sources
profile = CALL_API("fmp/profile", {"symbol": "AAPL"})
gdp = CALL_API("fred/series/observations", {"series_id": "GDP", "observation_start": "2015-01-01"})
print(profile)
print(gdp["observations"][-4:])

# Turn 3 (run_python): variables from turn 2 are still defined
interest = CALL_API("gt/google_trends_overtime", {"q": "iPhone", "geo": "US"})
print(interest)
```

The interpreter acts as a scratchpad for the whole call, so a complex analysis
can be built up over many `run_python` rounds; `max_interrupt_steps` on the
agent caps how many.

---

## Pattern 2: Explore hypotheses with dialog forks

Fork a dialog to evaluate several hypotheses from the same conversation state,
then bring the results back to the main thread:

```python
from lllm import Tactic


class HypothesisTactic(Tactic):
    name = "hypothesis_explorer"
    agent_group = ["analyst"]

    def call(self, task: str, hypotheses: list[str] = ()) -> str:
        analyst = self.agents["analyst"]

        # Establish shared context once
        analyst.open("base")
        analyst.receive(f"Background: {task}\nSummarize the key facts.")
        analyst.respond()

        results = {}
        for i, hypothesis in enumerate(hypotheses):
            analyst.fork("base", f"branch_{i}")   # each branch starts from the shared context
            analyst.receive(f"Evaluate this hypothesis: {hypothesis}")
            results[hypothesis] = analyst.respond().content
            analyst.close(f"branch_{i}")

        # Decide on the main thread, which never saw the branches
        analyst.switch("base")
        summary = "\n".join(f"- {h}: {r[:200]}" for h, r in results.items())
        analyst.receive(f"Evaluations:\n{summary}\n\nWhich hypothesis is strongest, and why?")
        return analyst.respond().content


explorer = HypothesisTactic({
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "analyst", "system_prompt": "You are a rigorous analyst."}],
})

print(explorer(
    "Weekly active users dropped 20% in Q3.",
    hypotheses=["a pricing change", "seasonality", "a broken onboarding flow"],
))
```

Extra keyword arguments to a tactic call, here `hypotheses`, are passed to
`call()`. Every branch shares the base context without paying for, or being
biased by, the other branches.

---

## Pattern 3: Typed multi-stage pipeline

Use parsers to pull structure out of one stage and Pydantic models as the
interface to the next:

```python
from pydantic import BaseModel

from lllm import Prompt, Tactic
from lllm.runtimes.native import DefaultTagParser


class Plan(BaseModel):
    steps: list[str]
    risks: list[str]


plan_prompt = Prompt(
    path="pipeline/plan",
    prompt=(
        "Create a plan for: {task}\n\n"
        "Write each step as <step>...</step> and each risk as <risk>...</risk>."
    ),
    parser=DefaultTagParser(xml_tags=["step", "risk"]),
)


class PlanAndExecute(Tactic):
    name = "plan_and_execute"
    agent_group = ["planner", "executor"]
    output_type = list[str]

    def call(self, task: str) -> list[str]:
        planner = self.agents["planner"]
        executor = self.agents["executor"]

        # Stage 1: a structured plan (the parser enforces the tags, with repairs)
        planner.open("plan")
        planner.receive_prompt(plan_prompt, {"task": task})
        tags = planner.respond().parsed["xml_tags"]
        plan = Plan(steps=tags["step"], risks=tags["risk"])

        # Stage 2: execute each step in its own dialog
        results = []
        for i, step in enumerate(plan.steps):
            executor.open(f"step_{i}")
            executor.receive(f"Known risks: {plan.risks}\n\nExecute this step: {step}")
            results.append(executor.respond().content)
            executor.close(f"step_{i}")
        return results


pipeline = PlanAndExecute({
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [
        {"name": "planner", "system_prompt": "You plan projects as short, concrete steps."},
        {"name": "executor", "system_prompt": "You carry out one step at a time and report the result."},
    ],
})

for result in pipeline("set up a benchmark for a text classifier"):
    print("-", result[:80])
```

The plan prompt is sent with `receive_prompt()` so its parser drives the turn,
and `output_type = list[str]` validates what the tactic returns.

---

## Pattern 4: High-throughput batches

Process many items concurrently with `bcall()` and record every call with an
observer:

```python
from lllm import Prompt, Tactic
from lllm.runtimes.native import DefaultTagParser

classify_prompt = Prompt(
    path="classifier/task",
    prompt=(
        "Classify the sentiment of this review as positive, negative, or neutral. "
        "Answer as <label>...</label>.\n\nReview: {review}"
    ),
    parser=DefaultTagParser(xml_tags=["label"]),
)


class ClassificationTactic(Tactic):
    name = "classifier"
    agent_group = ["classifier"]

    def call(self, task: str) -> str:
        if not task.strip():
            raise ValueError("empty review")
        classifier = self.agents["classifier"]
        classifier.open("classify")
        classifier.receive_prompt(classify_prompt, {"review": task})
        return classifier.respond().parsed["xml_tags"]["label"][0].strip()


classifier = ClassificationTactic(
    {
        "max_workers": 16,   # default thread count for bcall()
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [{"name": "classifier", "system_prompt": "You label sentiment."}],
    },
    observer=JsonlRunLog("batch_runs.jsonl"),
)

reviews = ["Great battery life.", "", "Arrived broken.", "It is fine."]

results = classifier.bcall(reviews, fail_fast=False, tags={"batch": "2026-03-17", "env": "prod"})
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"item {i} failed: {result}")
    else:
        print(f"item {i}: {result}")
```

- With `fail_fast=False`, failures come back as exception objects in place of
  results and the batch keeps going. The default, `fail_fast=True`, raises the
  first error.
- Every item, successful or not, lands in `batch_runs.jsonl` with its tags,
  cost, and traceback, ready for the queries from Lesson 7.
- Provider rate limits are retried with exponential backoff, up to the agent's
  `max_rate_limit_retries` (default 8).

---

## Pattern 5: Stream responses

Give an agent a `stream_handler` to receive text as it is generated:

```python
from lllm import Agent, Prompt
from lllm.runtimes.native.invokers import build_invoker
from lllm.runtimes.native.invokers.base import BaseStreamHandler


class PrintStreamHandler(BaseStreamHandler):
    def handle_chunk(self, chunk_content: str, chunk_response) -> None:
        print(chunk_content, end="", flush=True)


storyteller = Agent(
    name="storyteller",
    system_prompt=Prompt(path="storyteller/system", prompt="You are a storyteller."),
    model="gpt-4o",
    llm_invoker=build_invoker({"invoker": "litellm"}),
    stream_handler=PrintStreamHandler(),
)

storyteller.open("chat")
storyteller.receive("Tell me a short story about a robot.")
message = storyteller.respond()   # text prints as it arrives
print()
print(message.usage)              # usage is still recorded for streamed replies
```

`handle_chunk()` receives each text delta and the provider's raw chunk object.
The complete reply is still assembled, parsed, and appended to the dialog, so
parsers, repairs, and sessions behave exactly as without streaming.

---

## Pattern 6: Custom repair and tool-result messages

Each prompt carries a handler that writes the messages the agent loop sends on
your behalf. Reword them per prompt with `DefaultSimpleHandler`:

```python
from lllm import Prompt
from lllm.runtimes.native import DefaultSimpleHandler, DefaultTagParser

strict_answer = Prompt(
    path="strict/answer",
    prompt="Compute {expression}. Reply with only <answer>NUMBER</answer>.",
    parser=DefaultTagParser(xml_tags=["answer"]),
    handler=DefaultSimpleHandler(
        exception_msg="Your reply could not be parsed:\n{error_message}\nReply again using exactly the required format.",
        interrupt_msg="Tool output:\n{call_results}",
        interrupt_final_msg="Stop calling tools now and give your final answer.",
    ),
)
```

| Message | Sent when | Placeholder |
|---|---|---|
| `exception_msg` | the reply fails parsing or validation | `{error_message}` |
| `interrupt_msg` | a tool result is fed back | `{call_results}` (keep it, or the result is lost) |
| `interrupt_final_msg` | the tool-round limit is reached | none |

For strategies that adapt as attempts accumulate, subclass the handler. Each
hook receives the prompt driving the turn and the live `AgentCallSession`:

```python
from lllm import Prompt
from lllm.runtimes.native import AgentCallSession, DefaultSimpleHandler, DefaultTagParser


class EscalatingHandler(DefaultSimpleHandler):
    def on_exception(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        attempt = session.exception_retries_count
        advice = (
            "Reply with ONLY the required tags and nothing else."
            if attempt >= 2
            else "Please fix the format."
        )
        return prompt.extend(
            path=f"{prompt.path}/repair_{attempt}",
            prompt=f"Attempt {attempt} failed: {{error_message}}\n{advice}",
        )


escalating = Prompt(
    path="escalating/answer",
    prompt="Compute {expression}. Reply with <answer>NUMBER</answer>.",
    parser=DefaultTagParser(xml_tags=["answer"]),
    handler=EscalatingHandler(),
)
```

The prompt a hook returns drives the next model call, so build it with
`prompt.extend(...)`: the copy keeps the parser and tools. The session gives
you every earlier exception, tool round, and provider retry, so a handler can
change strategy over time; it could even call another agent to diagnose the
failure.

---

## Pattern 7: Image input

Agents accept images alongside text:

```python
agent.open("vision")
agent.receive_image("chart.png", caption="Q1 revenue chart")
agent.receive("What trends do you see in this chart?")
print(agent.respond().content)
```

`receive_image()` accepts a file path, a base64-encoded PNG, JPEG, WebP, or GIF,
or a Pillow `Image`, and raises `ValueError` for anything else. The model must
support image input.

---

## Pattern 8: Test tactics offline with a scripted invoker

Invokers are pluggable, so tests can replace the model with scripted replies
and run whole tactics without network access or API keys:

```python
from lllm import Message, Role, Tactic, register_invoker
from lllm.runtimes.native import InvokeResult
from lllm.runtimes.native.invokers.base import BaseInvoker


class ScriptedInvoker(BaseInvoker):
    """Return pre-written replies in order, parsing them like a real invoker."""

    def __init__(self, config):
        super().__init__(config)
        self.replies = list(config.get("replies", []))

    def call(self, dialog, model, model_args=None, parser_args=None, responder="assistant",
             metadata=None, api_type=None, stream_handler=None):
        content = self.replies.pop(0)
        prompt = dialog.top_prompt          # the prompt driving this turn
        parsed, errors = {}, []
        if prompt.parser is not None:
            try:
                parsed = prompt.parse(content, **(parser_args or {}))
            except Exception as exc:        # becomes a repair, as with a real model
                errors.append(exc)
        message = Message(role=Role.ASSISTANT, name=responder, content=content,
                          parsed=parsed, model=model)
        return InvokeResult(message=message, execution_errors=errors)


register_invoker("scripted", ScriptedInvoker)


class SummarizerTactic(Tactic):
    name = "summarizer"
    agent_group = ["writer"]

    def call(self, task: str) -> str:
        writer = self.agents["writer"]
        writer.open("session")
        writer.receive(f"Summarize: {task}")
        return writer.respond().content


tactic = SummarizerTactic({
    "invoker": "scripted",
    "invoker_config": {"replies": ["- first summary", "- second summary"]},
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "writer", "system_prompt": "You summarize."}],
})

assert tactic("document one") == "- first summary"
assert tactic("document two") == "- second summary"
```

- `register_invoker(name, builder)` makes a builder available to configs through
  the top-level `invoker` key; the builder receives `invoker_config`.
- One invoker is shared by all calls of a tactic instance, which is why the
  replies are consumed in order across calls.
- For an agent you built by hand, assign `agent.llm_invoker` directly.
- `lllm create` projects ship a smoke test built on the same idea.

---

## Architecture at a glance

```text
lllm.toml                  package declaration (Lesson 6)
├── prompts/               Prompt objects
├── configs/               agent configs (YAML)
├── tactics/               Tactic classes
├── tools/                 @tool functions
└── proxies/               BaseProxy classes

Runtime                    registry of prompts, tools, proxies, tactics, configs
Tactic                     the program: call(task) -> result, served as is over HTTP
└── Agent(s)               built from config: model, system prompt and tools, invoker, limits
    └── Dialog(s)          append-only, forkable conversation state
        └── Prompt         template + parser + tools + handlers (+ format), one per turn
Invoker                    talks to the model provider (LiteLLM by default)
TacticCallSession          per-call trace: agent and sub-tactic sessions, costs, errors, sent to the observer
```

---

## Putting it all together

When a project outgrows a single file, let LLLM lay it out:

```bash
lllm create my-app
```

```text
my-app/
├── lllm.toml          # [package] name = "my_app", plus the folders to discover
├── prompts/
│   └── assistant.py   # system = Prompt(path="system", ...), registered as assistant/system
├── configs/
│   └── default.yaml   # tactic_type: assistant, plus global and agent_configs
├── tactics/
│   └── assistant.py   # class AssistantTactic(Tactic): name = "assistant"; agent_group = ["assistant"]
├── main.py            # create_tactic(): load_runtime, resolve_config, build_tactic
└── tests/
    └── test_smoke.py  # builds the tactic without calling a model
```

The other templates (`--template pipeline`, `proxy`, `research`, `service`)
start from richer layouts. A production entry point loads the package, resolves
the config, and builds the tactic by name:

```python
from lllm import build_tactic, load_runtime, resolve_config

runtime = load_runtime("my-app/lllm.toml")
config = resolve_config("default", registry=runtime)
tactic = build_tactic(config, registry=runtime, observer=JsonlRunLog("production_runs.jsonl"))

results = tactic.bcall(
    ["first task", "second task"],
    max_workers=20,
    fail_fast=False,
    tags={"version": "1.2", "env": "prod"},
)
```

`load_runtime()` reads `lllm.toml`, registers every prompt, config, and tactic
in the package (it also becomes the default runtime), and `build_tactic()`
constructs the class named by `tactic_type` with its stable
`my_app::assistant` identity. [Lesson 6](06-packages-and-hub.md) explains the
package layout, configs with inheritance, dependencies, and publishing.

---

## What to explore next

- **Serving and composition**: serve tactics with `lllm serve` and
  `create_tactic_app` ([Lesson 5](05-native-tactics.md), Step 11), call them
  remotely, and compose them by reference ([Lesson 9](09-serving.md)).
- **Pydantic AI agents**: run Pydantic AI agents as tactics and give tactics to
  Pydantic AI agents as tools ([Lesson 10](10-pydantic-ai.md)).
- **Channels**: connect tactics through SSSN channels ([Lesson 11](11-channels.md)).
- **Agent skills**: the `skills` config key loads skills that follow the
  agentskills.io `SKILL.md` format from `.agents/skills/` or `.claude/skills/`,
  adds a short catalog to the system prompt, and gives the agent an
  `activate_skill` tool that loads a skill's full instructions on demand; like
  other agent tools, it is offered on every turn. See the
  [Agent reference](../core/agent.md).
- **Responses API and web search**: `api_type: response` on an agent selects
  the provider's Responses API, and `Prompt(addon_args={"web_search": True})`
  adds the provider's web search tool for that turn.
- **Custom invokers**: wrap any model API behind `BaseInvoker` and register it
  with `register_invoker` ([Invokers reference](../core/invokers.md)).
- **Computer use**: `lllm.runtimes.native.tools.cua` is an experimental
  browser-automation agent built on Playwright (install the `tools` extra).

**Next:** [9. Serving and Composition](09-serving.md)
