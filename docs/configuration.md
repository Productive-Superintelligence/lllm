# Configuration

A tactic's agents are described by a config: the model each agent uses, its
system prompt, its settings and tools, and how many retries it gets. The
config is data, a dict or a YAML file, so you can review, diff, and swap the
shape of a system without touching the program that runs it.

```python title="release_notes.py"
from lllm import Tactic


class ReleaseNotes(Tactic):
    name = "release_notes"
    agent_group = ["writer", "editor"]

    def call(self, changes: str) -> str:
        writer, editor = self.agents["writer"], self.agents["editor"]
        writer.open("draft")
        writer.receive(f"Write release notes for: {changes}")
        draft = writer.respond().content
        editor.open("edit")
        editor.receive(f"Tighten these release notes:\n{draft}")
        return editor.respond().content


config = {
    "global": {  # (1)!
        "model_name": "gpt-4o-mini",
        "model_args": {"temperature": 0.2},
    },
    "agent_configs": [  # (2)!
        {"name": "writer", "system_prompt": "You write release notes."},
        {
            "name": "editor",
            "system_prompt": "You edit for brevity.",
            "model_name": "gpt-4o",  # (3)!
            "model_args": {"max_tokens": 200},
        },
    ],
}

notes = ReleaseNotes(config)
for name, agent in notes.agents.items():
    print(name, agent.model, agent.model_args)
    #> writer gpt-4o-mini {'temperature': 0.2}
    #> editor gpt-4o {'temperature': 0.2, 'max_tokens': 200}
print(notes("faster startup; fixed the login timeout"))
#> Faster startup, and login no longer times out.
```

1. `global` holds the defaults for every agent.
2. One entry per agent in the tactic's `agent_group`, matched by `name`.
   An entry overrides `global`, and `model_args` dicts are merged.
3. One agent can use another model without touching the rest.

The config is read once, when the tactic is built, and every call gets
fresh agents made from it. `notes.agents` holds the agents built at
construction, which is a quick way to check what a config produces.

## Agent settings

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | The agent in `agent_group` this entry describes |
| `model_name` | required | A [LiteLLM model name](models.md), such as `gpt-4o-mini` |
| `system_prompt` | | The system prompt as text |
| `system_prompt_path` | | A registered [prompt](prompts.md), such as `analyst/system` |
| `model_args` | `{}` | Sent with every call: `temperature`, `max_tokens`, `api_base`, ... |
| `api_type` | `completion` | `response` selects OpenAI's [Responses API](models.md#chat-completions-or-responses) |
| `max_interrupt_steps` | 5 | Tool rounds per `respond()` |
| `max_exception_retry` | 3 | Repairs of replies that fail to parse |
| `max_llm_recall` | 0 | Retries after a provider error |
| `max_rate_limit_retries` | 8 | Retries after a rate limit |
| `tools` | `[]` | Registered tools, tactics, and proxies ([Tools](tools.md#giving-tools-to-agents)) |
| `proxy` | | API proxies and the code interpreter ([Proxies](proxies.md)) |
| `skills` | | Agent skills: local names, `SKILL.md` URLs, or `"*"` ([Agents](agents.md#tools-and-skills)) |
| `context_manager` | | Keeps the dialog inside the context window ([Dialogs](dialogs.md)) |
| `inference_hints` | | Session and priority hints ([Inference hints](inference-hints.md)) |

Every agent needs `model_name` and one of `system_prompt` or
`system_prompt_path`, from its own entry or from `global`. Both kinds of
system prompt can have `{placeholders}`, filled when the agent opens a
dialog with `open(alias, prompt_args={...})`. The limits are explained in
[the agent loop](agents.md#the-agent-loop).

A config is checked when the tactic is built, before any model call. An
agent without a model or a system prompt, or an agent in `agent_group`
without an entry, raises `ValueError` naming it, and a `system_prompt_path`
that is not registered raises `KeyError`. A key LLLM does not recognize is
added to `model_args`, with a warning that names it, so a typo such as
`temprature` shows up in the logs instead of vanishing.

## Global defaults

Each entry is merged onto `global`. Dicts merge key by key, while lists
and plain values replace. So one agent can add a model argument without
repeating the others, and an empty list removes a global one:

```yaml
global:
  model_name: gpt-4o-mini
  model_args: {temperature: 0.2}
  tools: [search, fetch_page]
agent_configs:
  - name: researcher          # both tools
    system_prompt: You find sources.
  - name: writer
    system_prompt: You write the report.
    model_args: {max_tokens: 800}  # and temperature 0.2
    tools: []                      # no tools
```

## Tactic-level keys

| Key | Read by | Meaning |
| --- | --- | --- |
| `global` | the tactic | Defaults for every agent |
| `agent_configs` | the tactic | One entry per agent |
| `base` | `resolve_config` | A config to inherit from ([Inheritance](#inheritance)) |
| `tactic_type` | `build_tactic` | The registered tactic to build ([Building by name](#building-a-tactic-by-name)) |
| `invoker` | the tactic | The [invoker](models.md#custom-invokers) that calls models (default `litellm`) |
| `invoker_config` | the tactic | Settings passed to the invoker (default: the whole config) |
| `max_workers` | `bcall`, `ccall` | How many calls run at once (default 4) |

Anything else at the top level is yours. The tactic keeps the whole config
as `self.config`, so `call()` can read a rubric, a word limit, or a list of
sources from it.

## Configs in a project

In a project, configs are YAML files and prompts are Python files, in
folders that the project's `lllm.toml` lists ([Packages](packages.md)
covers the manifest). `lllm create` makes this layout:

```text
markets/
├── lllm.toml
├── configs/default.yaml
├── prompts/analyst.py
└── tactics/report.py
```

This script writes the first three files into the current folder, with the
tactic in the script itself, and runs it:

```python title="project.py"
from pathlib import Path

from lllm import Tactic, load_runtime, resolve_config

Path("lllm.toml").write_text('[package]\nname = "markets"\n')
Path("prompts").mkdir()
Path("prompts/analyst.py").write_text("""
from lllm import Prompt

system = Prompt(path="system", prompt="You analyze {market} markets.")
""")
Path("configs").mkdir()
Path("configs/default.yaml").write_text("""
global:
  model_name: gpt-4o-mini
agent_configs:
  - name: analyst
    system_prompt_path: analyst/system
""")


class Report(Tactic):
    name = "report"
    agent_group = ["analyst"]

    def call(self, question: str) -> str:
        analyst = self.agents["analyst"]
        analyst.open("report", prompt_args={"market": "energy"})
        analyst.receive(question)
        return analyst.respond().content


runtime = load_runtime()  # (1)!
print(runtime.keys("config"))
#> ['markets.configs:default']
print(runtime.keys("prompt"))
#> ['markets.prompts:analyst/system']

report = Report(resolve_config("default"))  # (2)!
print(report("Why did oil prices fall this week?"))
#> Oil fell on rising inventories and weaker demand forecasts.
```

1. `load_runtime()` finds `lllm.toml` and registers what its folders hold,
   each under the package name.
2. `resolve_config()` returns the config as a dict. The tactic then loads
   `analyst/system` from the runtime.

Names follow the files:

- A config is named by its path under `configs/` without `.yaml`:
  `default`, `experiments/fast`.
- A prompt is named by its file and its `path`: `Prompt(path="system")` in
  `prompts/analyst.py` is `analyst/system`.
- A bare name looks in your own package. Use `pkg:name` for a dependency's
  config or prompt, such as `shared:default` or `shared:analyst/system`.

`load_runtime()` reads the file named by the `LLLM_CONFIG` environment
variable if it is set, and otherwise the first `lllm.toml` in the working
directory or its parents; `load_runtime("path/to/lllm.toml")` picks one
explicitly. It also loads the packages installed in the project's
`lllm_packages/` folder and in `~/.lllm/packages/`. Nothing is read when you
import `lllm`, and a YAML file is read the first time it is used.

## Building a tactic by name

`build_tactic(config)` constructs the registered tactic named by the
config's `tactic_type`, so the config chooses the program as well as the
agents:

```python title="by_name.py"
from lllm import Tactic, build_tactic, register_tactic_class


@register_tactic_class  # (1)!
class Translate(Tactic):
    name = "translate"
    agent_group = ["translator"]

    def call(self, text: str) -> str:
        translator = self.agents["translator"]
        translator.open("t")
        translator.receive(text)
        return translator.respond().content


config = {
    "tactic_type": "translate",
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "translator", "system_prompt": "Translate into French."}
    ],
}
tactic = build_tactic(config)
print(type(tactic).__name__, tactic("Good morning"))
#> Translate Bonjour
```

1. In a project, `load_runtime()` registers every `Tactic` subclass with a
   `name` it finds under `tactics/`. The decorator does the same by hand.

So a project starts with `load_runtime()` and then
`build_tactic(resolve_config("default"))`, which is what `lllm create` puts
in `main.py`. Pass `name="..."` to build another tactic with the same
config. Other keyword arguments, such as `observer=`, go to the tactic's
constructor.

## Inheritance

A config can name a `base` config and change only what differs. The merge
is the same as for `global`: dicts merge and everything else replaces.

```python title="inherit.py"
from pathlib import Path

from lllm import load_runtime, resolve_config

Path("lllm.toml").write_text('[package]\nname = "markets"\n')
Path("configs/experiments").mkdir(parents=True)
Path("configs/default.yaml").write_text("""
global:
  model_name: gpt-4o
  model_args: {temperature: 0.1}
agent_configs:
  - name: analyst
    system_prompt: You analyze markets.
""")
Path("configs/experiments/fast.yaml").write_text("""
base: default
global:
  model_name: gpt-4o-mini
  model_args: {max_tokens: 500}
""")

load_runtime()
config = resolve_config("experiments/fast")
print(config["global"]["model_name"])
#> gpt-4o-mini
print(config["global"]["model_args"])  # (1)!
#> {'temperature': 0.1, 'max_tokens': 500}
print([agent["name"] for agent in config["agent_configs"]])  # (2)!
#> ['analyst']
```

1. `model_args` is a dict, so the two files' arguments are merged.
2. `fast.yaml` has no `agent_configs`, so it keeps its base's list.

A base can have its own `base`, so experiments can stack. The result has no
`base` key, and a cycle raises `ValueError`. `load_config(name)` returns a
file as written, without following `base`.

A config meant to be shared from a package should name its parent with the
package prefix (`base: shared:default`), so it resolves from any project
that depends on it. To pin a dependency's config with your own changes, use
`vendor_config`, which resolves the chain and merges your overrides on top:

```python test="skip" reason="needs a package named shared"
from lllm.runtimes.native.core.config import vendor_config

config = vendor_config(
    "shared:default", {"global": {"model_name": "gpt-4o"}}
)
```

## API keys and environment variables

Configs name models; they do not hold secrets. Keys and paths come from
the environment:

| Variable | Read by | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ... | LiteLLM | Provider keys ([Models](models.md#api-keys)) |
| `LLLM_CONFIG` | `load_runtime()` | An `lllm.toml` file, or a folder with one, to load instead of searching |

`LLLM_CONFIG` lets one project run with another manifest, for example
`LLLM_CONFIG=deploy/lllm.prod.toml python main.py`. A package can also
declare the keys it needs: `lllm setup` asks for the missing ones and
stores them in a usable OS keyring when there is one, and in `.env.local`
otherwise, and `lllm serve` reads them from there. See
[Packages](packages.md#api-keys).
