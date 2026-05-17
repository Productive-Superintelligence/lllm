from lllm import Tactic


class ServiceTactic(Tactic):
    name = "service"
    agent_group = ["assistant"]

    def call(self, task: str) -> str:
        assistant = self.agents["assistant"]
        assistant.open("request")
        assistant.receive(task)
        return assistant.respond().content
