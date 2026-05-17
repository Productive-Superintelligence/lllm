from lllm import Prompt


researcher_system = Prompt(
    path="researcher/system",
    prompt=(
        "You are a careful research analyst for {{project_title}}. "
        "Identify useful angles, evidence needs, risks, and next experiments."
    ),
)

synthesizer_system = Prompt(
    path="synthesizer/system",
    prompt=(
        "You synthesize research notes into concise, decision-useful summaries. "
        "Separate findings, uncertainties, and next steps."
    ),
)

research_task = Prompt(
    path="tasks/research",
    prompt="Research this topic and produce concise notes:\n\n{topic}",
)

synthesis_task = Prompt(
    path="tasks/synthesize",
    prompt="Synthesize these research notes into a short brief:\n\n{notes}",
)
