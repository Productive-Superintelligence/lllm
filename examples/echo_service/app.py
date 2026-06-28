from lllm.services import create_tactic_app

from tactics import build_tactic

app = create_tactic_app(build_tactic())
