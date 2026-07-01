from lllm.services import create_tactic_app

from tactics import build_tactic


app = create_tactic_app(
    build_tactic(),
    title="LLLM Native Brief Service",
    description="Offline native runtime example served through the tactic API.",
)
