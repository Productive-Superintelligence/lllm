import json
from lllm import Orchestrator, Prompt, Function, FunctionCall
from lllm.core.models import register_prompt

# Define a tool
def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather in a given location"""
    return json.dumps({"location": location, "temperature": "25", "unit": unit, "condition": "Sunny"})

weather_tool = Function(
    name="get_weather",
    description="Get the current weather in a given location",
    properties={
        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    required=["location"]
)
weather_tool.link_function(get_weather)

# Define prompt with tool
tool_prompt = Prompt(
    path="tool_chat",
    prompt="You are a helpful assistant. User says: {user_input}",
    functions_list=[weather_tool]
)

def main():
    register_prompt(tool_prompt)
    
    config = {
        "name": "tool_agent",
        "log_dir": "./logs",
        "log_type": "localfile",
        "agent_configs": {
            "assistant": {
                "model_name": "gpt-4o-mini",
                "system_prompt_path": "tool_chat",
                "temperature": 0.0,
            }
        }
    }

    class ToolAgent(Orchestrator):
        agent_type = "tool_user"
        agent_group = ["assistant"]
        
        def call(self, task: str, **kwargs):
            dialog = self.agents["assistant"].init_dialog({"user_input": task})
            # The agent loop in Agent.call handles function execution automatically
            response, dialog, interrupts = self.agents["assistant"].call(dialog)
            return response.content

    agent = ToolAgent(config, ckpt_dir="./ckpt")
    response = agent("What is the weather in Tokyo?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
