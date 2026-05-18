from lllm import Prompt


planner_system = Prompt(
    path="planner/system",
    prompt=(
        "You are a practical planner for {{project_title}}. "
        "Break the user's goal into concise, ordered work items."
    ),
)

writer_system = Prompt(
    path="writer/system",
    prompt=(
        "You are a clear technical writer for {{project_title}}. "
        "Turn a plan into a useful draft with concrete details."
    ),
)

reviewer_system = Prompt(
    path="reviewer/system",
    prompt=(
        "You are a critical reviewer for {{project_title}}. "
        "Improve clarity, correctness, and actionability without adding fluff."
    ),
)

plan_task = Prompt(
    path="tasks/plan",
    prompt="Create a short execution plan for this goal:\n\n{task}",
)

write_task = Prompt(
    path="tasks/write",
    prompt="Use this plan to produce a complete draft:\n\n{plan}",
)

review_task = Prompt(
    path="tasks/review",
    prompt="Review and polish this draft. Return only the improved draft:\n\n{draft}",
)
