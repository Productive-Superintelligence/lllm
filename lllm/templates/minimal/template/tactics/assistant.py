from lllm import Tactic


class AssistantTactic(Tactic):
    name = "assistant"
    agent_group = ["assistant"]

    def call(self, task: str) -> str:
        assistant = self.agents["assistant"]
        assistant.open("chat")
        assistant.receive(task)
        return assistant.respond().content
