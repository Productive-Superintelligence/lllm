from enum import Enum
from typing import Any, Dict

from lllm.llm import Orchestrator, Prompts, build_agent


class AgentType(Enum):
    VANILLA = 'vanilla'


class Vanilla(Orchestrator):
    agent_type = AgentType.VANILLA.value
    agent_group = ['vanilla']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent = self.agents['vanilla']
        self.prompts = Prompts('vanilla')

    def call(self, task: str, **kwargs):
        dialog = self.agent.init_dialog()
        self.st.write(task)
        message = self.agent.send_message(dialog, self.prompts('task_query'), {'task': task})
        self.st.write(message.content)
        response, dialog, _ = self.agent.call(dialog)
        dialog.overview(remove_tail=True, stream=self.st)
        return response.parsed or {'output': response.content}

