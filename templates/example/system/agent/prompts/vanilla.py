from lllm.core.prompt import Prompt
from pydantic import BaseModel




class TaskResponse(BaseModel):
    output: str


main_system_prompt = Prompt(
    path='system',
    prompt='System prompt for the agent with role: {role}'
)

task_query_prompt = Prompt(
    path='task_query',
    prompt='''
    Input the task description here {task}.
    ''',
    format=TaskResponse
)


