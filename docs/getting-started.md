# Getting Started

## Installation

Install LLLM using pip:

```bash
pip install lllm
```

## Basic Usage

Here is a simple example of how to create a chat agent:

```python
from lllm import Orchestrator, Prompt, register_prompt

# 1. Define a Prompt
simple_prompt = Prompt(
    path="simple_chat",
    prompt="You are a helpful assistant. User says: {user_input}"
)
register_prompt(simple_prompt)

# 2. Define an Agent
class SimpleAgent(Orchestrator):
    agent_type = "simple"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        # Initialize dialog with user input
        dialog = self.agents["assistant"].init_dialog({"user_input": task})
        # Call the agent
        response, dialog, _ = self.agents["assistant"].call(dialog)
        return response.content

# 3. Configure and Run
config = {
    "name": "simple_chat_agent",
    "log_dir": "./logs",
    "log_type": "localfile",
    "provider": "openai",
    "auto_discover": True,
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

Set `provider` to any backend you register via `lllm.providers.register_provider`, and flip `auto_discover` to `False` if you want to opt out of scanning `lllm.toml` paths when instantiating agents or proxies.

## Next Steps

- Learn about [Agents](core/agents.md)
- Explore [Prompts](core/prompts.md)
- Check out [Examples](https://github.com/yourusername/lllm/tree/main/examples)

## Sandbox Smoke Test

Want to verify that the Jupyter sandbox works in your environment without writing code? Run:

```bash
python examples/jupyter_sandbox_smoke.py --session demo
```

This creates a notebook under `.lllm_sandbox/demo/demo.ipynb` with a sample markdown and code cell. Open it in Jupyter or VS Code to inspect the generated notebook and build from there.

## Sample `lllm.toml`

Auto-discovery is easiest to learn by copying the example config:

```bash
cp lllm.toml.example lllm.toml
```

The sample points to `examples/autodiscovery/prompts/` and `examples/autodiscovery/proxies/`. Edit those folders (or add new ones) to register your own prompts and proxies automatically on import.

## Developer Test Suite

If you’re contributing to LLLM itself, run the full suite before sending a PR:

```bash
pytest
```

This command covers unit tests, sandbox mocks, CLI/template smoke tests, and the recorded OpenAI tool-call flows. When you need to refresh the recordings, update the JSON files under `tests/integration/recordings/` and rerun the integration tests.
