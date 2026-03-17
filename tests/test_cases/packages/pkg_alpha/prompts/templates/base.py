"""Base template prompts — reusable building blocks."""
from lllm.core.prompt import Prompt, DefaultSimpleHandler

system_base = Prompt(
    path="system_base",
    prompt="You are a helpful AI assistant. Be concise and accurate.",
    metadata={"type": "system", "version": "1.0"},
)

user_task = Prompt(
    path="user_task",
    prompt="Please help me with the following task:\n\n{task}\n\nContext: {context}",
    metadata={"type": "user_task", "version": "1.0"},
)

with_custom_handler = Prompt(
    path="with_custom_handler",
    prompt="Answer using available tools: {question}",
    handler=DefaultSimpleHandler(
        exception_msg="Error: {error_message}. Please try a different approach.",
        interrupt_msg="Tool results:\n{call_results}\n\nContinue your analysis.",
        interrupt_final_msg="Based on the tool results, provide your final answer.",
    ),
)

structured_output = Prompt(
    path="structured_output",
    prompt="Process {input} and return the result.",
    metadata={"output_format": "json"},
)
