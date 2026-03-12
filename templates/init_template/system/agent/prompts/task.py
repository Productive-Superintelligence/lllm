from lllm.core.prompt import Prompt

"""Initial prompts for the scaffolded agent."""

main_system_prompt = Prompt(
    path="{{project_name}}/system",
    prompt=(
        "You are the {{project_name}} agent. "
        "Answer with clear, concise explanations and note any assumptions."
    ),
)

task_query_prompt = Prompt(
    path="{{project_name}}/task_query",
    prompt=(
        "User task:\n"
        "{task}\n\n"
        "Respond using bullet points when possible."
    ),
)
