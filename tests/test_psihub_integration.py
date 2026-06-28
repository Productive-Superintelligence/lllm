from pydantic import BaseModel

from lllm import Tactic, endpoint
from lllm.integrations import tactic_resource


class ActInput(BaseModel):
    text: str


class ActOutput(BaseModel):
    text: str


class PolicyTactic(Tactic[ActInput, ActOutput]):
    name = "policy"
    input_type = ActInput
    output_type = ActOutput

    def _run(self, input_value, *, context=None):
        return ActOutput(text=input_value.text)

    @endpoint.post("/act", tags=("policy",))
    async def act(self, input_value, *, context=None):
        return await self.arun(input_value, context=context)


def test_tactic_resource_includes_custom_endpoint_metadata():
    resource = tactic_resource(PolicyTactic())

    assert resource["name"] == "policy"
    assert resource["endpoints"] == [
        {
            "name": "act",
            "method": "POST",
            "path": "/act",
            "mode": "run",
            "description": "",
            "tags": ["policy"],
        }
    ]
