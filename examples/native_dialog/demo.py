from lllm.runtimes.native import Dialog, FunctionCall, Prompt, Role, tool


@tool(description="Create a display label for a project area")
def label(topic: str) -> str:
    return topic.strip().title()


SYSTEM_PROMPT = Prompt(
    path="planner/system",
    prompt="You are a {style} planning assistant.",
)
TASK_PROMPT = Prompt(
    path="planner/task",
    prompt="Plan the next checkpoint for {project}.",
    function_list=[label],
)


def build_dialog() -> Dialog:
    dialog = Dialog(owner="planner")
    dialog.put_prompt(
        SYSTEM_PROMPT,
        prompt_args={"style": "careful"},
        role=Role.SYSTEM,
        name="system",
    )
    dialog.put_prompt(
        TASK_PROMPT,
        prompt_args={"project": "LLLM v2"},
        name="operator",
    )

    call = TASK_PROMPT.get_function("label")(
        FunctionCall(name="label", arguments={"topic": "native core"})
    )
    dialog.put_text(
        call.result_str or "",
        role=Role.TOOL,
        name="label",
        metadata={"function_call": call.model_dump(mode="json")},
    )
    return dialog


def build_retry_dialog() -> Dialog:
    return build_dialog().fork(last_n=1, first_k=1)


if __name__ == "__main__":
    print(build_dialog().overview())
