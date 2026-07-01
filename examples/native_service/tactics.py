from typing import Any

from pydantic import BaseModel, Field

from lllm.runtimes.native import Dialog, NativeTacticAdapter, Prompt, Role, Tactic


class BriefInput(BaseModel):
    topic: str
    audience: str = "maintainers"


class TranscriptEntry(BaseModel):
    role: str
    name: str
    content: str


class BriefOutput(BaseModel):
    title: str
    bullets: list[str] = Field(min_length=1)
    trace_id: str | None = None
    transcript: list[TranscriptEntry]


SYSTEM_PROMPT = Prompt(
    path="native-service/system",
    prompt="You write {tone} implementation notes for {audience}.",
)


class OfflineBriefNative(Tactic):
    """Offline native tactic that records prompt/dialog state before serving."""

    name = "native-brief"
    input_model = BriefInput
    output_model = BriefOutput

    def call(
        self,
        task: BriefInput,
        *,
        context: Any = None,
        tone: str = "concise",
    ) -> BriefOutput:
        dialog = Dialog(owner=self.name)
        dialog.put_prompt(
            SYSTEM_PROMPT,
            prompt_args={"tone": tone, "audience": task.audience},
            role=Role.SYSTEM,
            name="system",
        )
        dialog.put_text(f"Prepare a brief about {task.topic}.", name="operator")

        bullets = [
            f"Topic: {task.topic}",
            f"Audience: {task.audience}",
            "Runtime: native prompt and dialog state behind a protocol tactic.",
        ]
        dialog.put_text(
            "\n".join(f"- {bullet}" for bullet in bullets),
            role=Role.ASSISTANT,
            name=self.name,
        )

        trace_id = getattr(context, "trace_id", None)
        return BriefOutput(
            title=f"{task.topic.title()} Brief",
            bullets=bullets,
            trace_id=trace_id,
            transcript=[
                TranscriptEntry(
                    role=_role_value(message.role),
                    name=message.name,
                    content=message.content,
                )
                for message in dialog.messages
            ],
        )


def build_native_tactic() -> OfflineBriefNative:
    return OfflineBriefNative()


def build_tactic() -> NativeTacticAdapter:
    return NativeTacticAdapter(
        build_native_tactic(),
        description="Serve an offline native prompt/dialog workflow.",
        package_ref="psi://demo/native-service/tactics/native-brief",
        run_kwargs={"tone": "precise"},
        metadata={"example": "native-service"},
    )


def _role_value(role: Any) -> str:
    return role.value if hasattr(role, "value") else str(role)
