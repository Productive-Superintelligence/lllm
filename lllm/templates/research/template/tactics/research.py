from lllm import Tactic, load_prompt


class ResearchTactic(Tactic):
    name = "research"
    agent_group = ["researcher", "synthesizer"]

    def call(self, topic: str) -> str:
        researcher = self.agents["researcher"]
        synthesizer = self.agents["synthesizer"]

        researcher.open("research")
        researcher.receive_prompt(load_prompt("research/tasks/research"), {"topic": topic})
        notes = researcher.respond().content

        synthesizer.open("synthesis")
        synthesizer.receive_prompt(load_prompt("research/tasks/synthesize"), {"notes": notes})
        return synthesizer.respond().content
