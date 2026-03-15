# Agents

Agents are the core actors in the LLLM framework. They encapsulate the logic for interacting with LLMs, managing state (Dialog), and executing tools.

## The `Agent` Class

The `Agent` class represents a single LLM entity with a specific role, system prompt, and model configuration.

### Key Attributes

- `name`: The name or role of the agent (e.g., "assistant", "coder").
- `system_prompt`: The `Prompt` object defining the agent's persona and capabilities.
- `model`: The LLM model to use (e.g., "gpt-4o").
- `llm_provider`: The provider instance (e.g., `OpenAIProvider`).

## The `Orchestra` Class

`Orchestra` is the base class for building complex agentic systems. It allows you to compose multiple `Agent` instances and define custom orchestration logic.

### Creating a Custom Agent

To create a custom agent, subclass `Orchestra` and implement the `call` method:

```python
from lllm import Orchestra

class MyAgent(Orchestra):
    agent_type = "my_agent"
    agent_group = ["worker", "reviewer"] # Define sub-agents

    def call(self, task: str, **kwargs):
        # Implement your logic here
        pass
```
