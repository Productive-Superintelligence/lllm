from lllm import Prompt


system = Prompt(
    path="system",
    prompt=(
        "You are a concise assistant for the {{project_title}} project. "
        "Answer clearly and ask for missing details when the task is ambiguous."
    ),
)
