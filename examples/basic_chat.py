import os
from lllm import Agent, Prompt, Roles

# Define a simple prompt
# In a real app, this would be in a separate file and auto-discovered
simple_prompt = Prompt(
    path="simple_chat",
    prompt="You are a helpful assistant. User says: {user_input}"
)

def main():
    # Configuration usually comes from lllm.toml, but here we mock it
    config = {
        "name": "simple_chat_agent",
        "log_dir": "./logs",
        "log_type": "localfile",
        "agent_configs": {
            "assistant": {
                "model_name": "gpt-4o-mini",
                "system_prompt_path": "simple_chat", # referencing the path above
                "temperature": 0.7,
            }
        }
    }

    # Register the prompt manually since we are not using auto-discovery from files here
    from lllm.core.models import register_prompt
    register_prompt(simple_prompt)

    # Initialize Agent
    # Note: Orchestrator expects a class structure usually, but we can use the Agent class directly for simple cases
    # or use the build_agent factory if we had a proper Agent class defined.
    # Here we demonstrate direct usage of the internal Agent class for simplicity, 
    # although the framework encourages using Orchestrator subclasses.
    
    # Let's define a minimal Agent class
    from lllm import Orchestrator
    
    class SimpleAgent(Orchestrator):
        agent_type = "simple"
        agent_group = ["assistant"]
        
        def call(self, task: str, **kwargs):
            dialog = self.agents["assistant"].init_dialog({"user_input": task})
            response, dialog, _ = self.agents["assistant"].call(dialog)
            return response.content

    # Build the agent
    agent = SimpleAgent(config, ckpt_dir="./ckpt")
    
    # Run
    response = agent("Hello, how are you?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
