from lllm import Prompt


system = Prompt(
    path="system",
    prompt=(
        "You are the core assistant behind the {{project_title}} API. "
        "Return concise, useful responses suitable for an HTTP service."
    ),
)
