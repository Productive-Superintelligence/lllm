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

LLLM is a lightweight framework for developing **advanced agentic systems**. Allows users to build a complex agentic system with <100 LoC. Prioritizing minimalism, modularity, and reliability, it is specifically suitable for complex and frontier agentic systems beyond daily chat. While these fields require deep architectural customization and highly diverse demands, developers and researchers often face the burden of managing low-level complexities such as exception handling, output parsing, and API error management. LLLM bridges this gap by offering necessary abstractions that balance high-level encapsulation with the simplicity required for flexible experimentation. It also tries to make the code plain, compact, easy-to-understand, with less unnecessary indirection, thus easy for customization for different projects' needs, to allow researchers and developers to focus on the core research questions. See https://lllm.one for detailed documentation.


## Design Philosophy

It's more designed for developers and researchers, and is designed to be Pythonic like PyPI/npm, PyTorch, and Hugging Face, to help them build their own agentic systems easily, and to allow the modules built to be shared and reused by others. 

- **Agentic System as a Program**:  An agentic system = agents (≈ system prompt + base model, the “callers”) + prompts (the "functions" or "calls") + the tactics (the "program" that wires the callers and functions). The agent call makes LLM agents "deterministic" callers, for minimizing side effects, maximizing compositionality, and parallelizability.
- **Dialog as Internal "Mental" State**: Dialog is the "internal mental state" of each agent due to system prompt, different bodies in a talk maintain their own internal dialog, i.e., dialog is what each agent "sees" from the others, its objective, not subjective. Dialog is also a function stack for each agent, where top_prompt is the calling convention for the next turn.
- **Configuration as Declaration**: System shape is declared in config, not hardcoded. What exists (prompts, proxies) and how it's wired (agent configs) are expressed as data.

### Advanced Features
- **Tool calling as Programming**: Besides regular tool calling provided by LLMs, LLLM provides a mini-interpreter mode that calls tools as a mini Python script, and tools are wrapped through the proxy system, which is more flexible and powerful.
- **Tactics as Shared Library**: Tactics are the reusable library for different agentic systems, ideally you can "import" a tactic from a library, and use it in your own agentic system. Each prompt, proxy, config, and tactic is a "independent" module, and they are loaded on demand through the lllm.toml file. Theoretically, you can share your tactics, prompts, proxies, configs, etc. with others, and they can use them in their own agentic systems.
- **Replayable Logging**: The logging system is designed to be replayable, which means you can replay the logging to get the exact same result as the original run, and save the entire traces and costs, etc. This is useful for debugging, A/B testing, prompt optimization, etc.

The LLLM stops at tactic as its highest level of abstraction, i.e., low-level, for the higher levels, like the system itself, the system of systems, and the network of systems, etc., please refer to the [Simple System of Systems Network (SSSN) framework](https://github.com/Productive-Superintelligence/sssn) for more details.



## Installation

```bash
pip install lllm-core
```


## Quick Start

### Basic Chat

TODO

## Examples

Check `examples/` for more usage scenarios:

TODO

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

TODO 

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Experimental Features

- **Computer Use Agent (CUA)** – `lllm.tools.cua` offers browser automation via Playwright and the OpenAI Computer Use API. It is still evolving and may change without notice.
- **Responses API Routing** – opt into OpenAI’s Responses API by setting `api_type = "response"` per agent. This enables native web search/computer-use tools but currently targets OpenAI only.
- **Skills (WIP)** – For defining more complex base agents.


# Roadmap

## v0.1.0 Refactoring 
- [x] Refactor providers system: LiteLLM invoker (invokers/)
- [x] Refactor registry to runtime (runtime.py), and discovery system (discovery.py)
- [x] Refactor prompt model and prompt management (prompt.py)
  - [x] Prompt composition and inheritance
  - [x] More graceful tool (link_function)
  - [x] Clearing up ad-hoc designs
  - [x] Better parsing system, more intuitive argument passing
  - [x] Better handling system for error, exception, interrupt
- [x] Refactor message and dialog model/state management, better arg passing (dialog.py)
- [x] Refactor agent model, agent call (agent.py)
- [x] Refactor tactics (tactic.py)
- [x] Refactor config and package system (config.py, lllm.toml, etc.)
  - [x] Package system with `lllm.toml` — namespaced resource URLs (`pkg.section:resource`)
  - [x] Dependency tree with recursive loading and cycle detection
  - [x] Alias support (`as` for packages, `under` for virtual folder prefixes)
  - [x] Unified ResourceNode-based registry with lazy loading
  - [x] Named runtimes (`load_runtime`, `get_runtime`) for parallel experiments
  - [x] Auto-initialization from project root `lllm.toml`
  - [x] Agent config YAML: `global` defaults, `agent_configs` list, `base` inheritance with deep merge
  - [x] `AgentSpec` with inline `system_prompt` or `system_prompt_path` resolution
  - [x] `resolve_config()` for recursive config inheritance
  - [x] Convenience loaders: `load_prompt`, `load_tactic`, `load_proxy`, `load_config`, `load_resource`
- [x] Logger (cli logging), replayable logging system, and printing system (log.py, utils.py)
  - [x] `LogStore` with pluggable backends (`LocalFileBackend`, `SQLiteBackend`, `NoOpBackend`)
  - [x] Tag-based indexing and filtering, cost aggregation, export helpers
  - [x] Stable `pkg::name` tactic identity independent of file layout and aliases
  - [x] `ColoredFormatter` and `setup_logging` for terminal output
  - [x] Convenience factories: `local_store`, `sqlite_store`, `noop_store`
- [ ] Fast mode, 5-line code to build a simple system with no configuration.


## TODOs

- [ ] Proxy-based tool-calling, mini in-dialog interpreter (proxies/)
- [ ] Analysis tools based on the logging system, e.g., cost analysis, dialog analysis, etc. Basically, a GUI for the logging DB, and exporting an app with default dashboards using like Streamlit, Dash, Panel, etc.
- [ ] Support skills in agent config, see https://agentskills.io
- [ ] Default Context Manager for prune over-size dialogs
- [ ] Better sandbox, e.g., browser sandbox, code sandbox, etc. maybe use sandbox wheels (sandbox/)
- [ ] Tactics, Prompts, Proxies, Configs, etc. sharing system.


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

