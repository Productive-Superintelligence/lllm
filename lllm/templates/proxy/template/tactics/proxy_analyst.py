from lllm import Tactic


class ProxyAnalystTactic(Tactic):
    name = "proxy_analyst"
    agent_group = ["analyst"]

    def call(self, task: str) -> str:
        analyst = self.agents["analyst"]
        analyst.open("analysis")
        analyst.receive(task)
        return analyst.respond().content
