<div align="center">
  <img src="https://raw.githubusercontent.com/ChengJunyan1/lllm/main/assets/LLLM-logo.png" alt="LLLM Logo" width="600"/>
  <br>
  <h1>Low-Level Language Model (LLLM) </h1>
  <h4>Lightweight framework for building complex agentic systems</h4>
</div>
<p align="center">
  <a href="https://lllm.one">
    <img alt="Docs" src="https://img.shields.io/badge/API-docs-red">
  </a>
  <a href="https://github.com/chengjunyan1/lllm/tree/main/examples">
    <img alt="Examples" src="https://img.shields.io/badge/API-examples-994B00">
  </a>
  <a href="https://pypi.org/project/lllm-core/">
    <img alt="Pypi" src="https://img.shields.io/pypi/v/lllm-core.svg">
  </a>
  <a href="https://github.com/chengjunyan1/lllm/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/chengjunyan1/lllm">
  </a>
  <a href="https://discord.gg/aTah8r7YpM">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=LLLM&color=%235B65E9">
  </a>
</p>

LLLM is a lightweight framework designed to facilitate the rapid prototyping of advanced agentic systems. Allows users to build a complex agentic system with <100 LoC. 
Prioritizing minimalism, modularity, and type safety, it is specifically optimized for research in program synthesis and neuro-symbolic AI. 
While these fields require deep architectural customization, researchers often face the burden of managing low-level complexities such as exception handling, output parsing, and API error management. 
LLLM bridges this gap by offering necessary abstractions that balance high-level encapsulation with the simplicity required for flexible experimentation.
It also tries to make the code plain, compact, easy-to-understand, with less unnecessary indirection, thus easy for customization for different projects' needs, to allow researchers to focus on the core research questions. See https://lllm.one for detailed documentation.



## Design Thoughts

- Functional Design: Agent as a function — parsing defines the return type, the agent call loop enforces it.
- Dialog Tree as State: Dialog is the shared workspace (blackboard) that agents, tools, and users collectively build. top_prompt is the calling convention for the next turn. Forking creates branches for speculation and recovery.
- Declarative Design: System shape is declared in config, not hardcoded. What exists (prompts, proxies) and how it's wired (agent configs) are expressed as data.






## Installation

```bash
pip install lllm-core
```

## Features

- **Modular Architecture**: Core abstractions, providers, tools, and memory are decoupled.
- **Type Safety**: Built on Pydantic for robust data validation and strict typing.
- **Provider Interface**: First-class OpenAI support with an extensible interface for adding more providers as needed.
- **Neuro-Symbolic Design**: Advanced prompt management with structured output, exception handling, and interrupt logic.
- **API Proxies**: Secure code execution of external APIs for program synthesis.


## Quick Start

### Basic Chat

```python
from lllm import Orchestra, Prompt, register_prompt

# Define a prompt
simple_prompt = Prompt(
    path="simple_chat",
    prompt="You are a helpful assistant. User says: {user_input}"
)
register_prompt(simple_prompt)

# Define an Agent
class SimpleAgent(Orchestra):
    agent_type = "simple"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        dialog = self.agents["assistant"].init_dialog({"user_input": task})
        response, dialog, _ = self.agents["assistant"].call(dialog)
        return response.content

# Configure and Run
config = {
    "name": "simple_chat_agent",
    "log_dir": "./logs",
    "log_type": "localfile",
    "provider": "openai",           # or any provider registered via lllm.providers
    "auto_discover": True,          # set False to skip automatic prompt/proxy discovery
    "agent_configs": {
        "assistant": {
            "model_name": "gpt-4o-mini",
            "system_prompt_path": "simple_chat",
            "temperature": 0.7,
        }
    }
}

agent = SimpleAgent(config, ckpt_dir="./ckpt")
print(agent("Hello!"))
```

`provider` selects a registered backend (default `openai`), while `auto_discover` controls whether LLLM scans the paths listed in `lllm.toml` for prompts and proxies each time you spin up an agent or proxy.


## Examples

Check `examples/` for more usage scenarios:
- `examples/basic_chat.py`
- `examples/tool_use.py`
- `examples/proxy_catalog.py`
- `examples/jupyter_sandbox_smoke.py`

### Proxies & Tools

Built-in proxies (financial data, search, etc.) register automatically when their modules are imported. If you plan to call `Proxy()` directly, either:

1. Set up an `lllm.toml` with a `[proxies]` section so discovery imports your proxy folders on startup, or
2. Call `load_builtin_proxies()` to import the packaged modules, or manually import the proxies you care about (e.g., `from lllm.proxies.builtin import exa_proxy`).

This mirrors how prompts are auto-registered via `[prompts]` in `lllm.toml`.

Once proxies are loaded you can check what is available by calling `Proxy().available()`.

### Auto-Discovery Config

A starter `lllm.toml.example` lives in the repo root. Copy it next to your project entry point and edit the folder paths:

```bash
cp lllm.toml.example lllm.toml
```

The sample configuration points to `examples/autodiscovery/prompts/` and `examples/autodiscovery/proxies/`, giving you a working prompt (`examples/hello_world`) and proxy (`examples/sample`) to experiment with immediately.

## Testing & Offline Mocks

- Run the full suite (for framework developers): `pytest`.
- For an end-to-end agent/tool flow without real OpenAI requests, see `tests/integration/test_tool_use_mock_openai.py`. It uses the scripted client defined in `tests/helpers/mock_openai.py`, mirroring what a VCR fixture would capture.
- Want template smoke tests? `tests/integration/test_cli_template.py` runs `python -m lllm.cli create --name demo --template init_template` inside a temp directory.
- When you want parity with real OpenAI traffic, capture responses into JSON (see `tests/integration/recordings/sample_tool_call.json`) and point `load_recorded_completions` at your file. `tests/integration/test_tool_use_recording.py` shows how to replay those recordings without network access.
- Need an opt-in live OpenAI smoke test? Everything under `tests/realapi/` hits the actual APIs whenever `OPENAI_API_KEY` is present (e.g., `pytest tests/realapi/`). If the key is missing, pytest prints a notice and skips those tests, leaving the default mock-based suite as-is.
- Optional future work: keep capturing real-provider recordings as APIs evolve, and consider running `examples/jupyter_sandbox_smoke.py` in CI to validate notebook tooling automatically.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Experimental Features

- **Computer Use Agent (CUA)** – `lllm.tools.cua` offers browser automation via Playwright and the OpenAI Computer Use API. It is still evolving and may change without notice.
- **Responses API Routing** – opt into OpenAI’s Responses API by setting `api_type = "response"` per agent. This enables native web search/computer-use tools but currently targets OpenAI only.


# Roadmap

## v0.1.0 Refactoring 
- [x] Refactor providers system: LiteLLM invoker (invokers/)
- [x] Refactor registry to runtime (runtime.py), and discovery system (discovery.py)
- [x] Refactor prompt model and prompt management (prompt.py)
  - [x] Prompt composition and inheritance (compose templates, config tools, etc.), using jinja or something else like dspy solutions
  - [x] More graceful tool, i.e., now it relies on `link_function`, which separates declaration and definition of a tool, if it is good?
  - [x] Clearing up ad-hoc designs, like now cua args, etc. are nakedly attatched as properties, whenever there is a new feature, it is added as a property, which is not good.
  - [x] Better parsing system, more intuitive and better argument passing, now its through a dict, which is primitive and not type safe.
  - [x] Better handling system for error, exception, interrupt, etc. (need to be co-designed with the agent parts)
  --- 
  - [-] Provide a version control solution, either allow using git or something else, to assist tuning, A/B testing, prompt optimization, etc.
    - [-] Assisting tracking system to help experiment (may also need to be co-designed with the agentic system parts)
    - [-] we do not need to build those wheels in lllm, but should be friendly to integrate with other version control or management systems. May need to refer to these solutions, so knowing how to be friendly to them. Or directly provide an interface to a popular solution (which one is the most popular?). Like having an abstract base class for all such version control, management, analysis, etc.
    - Just do not do it, any design now will be an overly pre design
- [x] Refactor message and dialog model/state management, better arg passing (dialog.py)
  - Dialog provides low-level operations, and advanced arg passings are provided by Agent etc. below
- [x] Refactor agent model, agent call (agent.py)
- [ ] -> Refactor orchestrator and config system (agent.py, config.py, lllm.toml)
- [ ] Proxy-based tool-calling, mini in-dialog interpreter (proxies/, sandbox/)
- [ ] Logger (cli logging), replayable logging system (log.py)

## Future Roadmap
- [ ] Gradient mode for tuning/training


<!-- 
git status          # ensure no stray files you don’t want in the sdist
rm -rf dist build *.egg-info    # clean

python -m build     # creates dist/lllm-<version>.tar.gz and .whl

# test locally
python -m venv /tmp/lllm-release
source /tmp/lllm-release/bin/activate
pip install dist/lllm-<version>-py3-none-any.whl
python -c "import lllm; print(lllm.__version__)"
deactivate

# upload
python -m twine upload dist/*

# push tag
git tag -a v0.0.1.3 -m "Release 0.0.1.3"
git push origin main --tags 


# update doc
mkdocs build --strict
mkdocs gh-deploy --force --clean

-->

