from enum import Enum
from typing import Any, Dict

from lllm.llm import Orchestrator, Prompts, build_agent

PROMPT_ROOT = "{{project_name}}"


class AgentType(Enum):
    VANILLA = 'vanilla'


class Vanilla(Orchestrator):
    agent_type = AgentType.VANILLA.value
    agent_group = ['vanilla']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent = self.agents['vanilla']
        self.prompts = Prompts(PROMPT_ROOT)

    def call(self, task: str, **kwargs):
        dialog = self.agent.init_dialog()
        self.st.write(task)
        dialog.send_message(self.prompts('task_query'), {'task': task})
        response, dialog, _ = self.agent.call(dialog)
        dialog.overview(remove_tail=True, stream=self.st)
        return response.parsed or {'output': response.content}

