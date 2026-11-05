from lllm import Tactic, load_prompt


class PipelineTactic(Tactic):
    name = "pipeline"
    agent_group = ["planner", "writer", "reviewer"]

    def call(self, task: str) -> str:
        planner = self.agents["planner"]
        writer = self.agents["writer"]
        reviewer = self.agents["reviewer"]

        planner.open("plan")
        planner.receive_prompt(load_prompt("pipeline/tasks/plan"), {"task": task})
        plan = planner.respond().content

        writer.open("draft")
        writer.receive_prompt(load_prompt("pipeline/tasks/write"), {"plan": plan})
        draft = writer.respond().content

        reviewer.open("review")
        reviewer.receive_prompt(load_prompt("pipeline/tasks/review"), {"draft": draft})
        return reviewer.respond().content
