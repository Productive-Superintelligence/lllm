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

    @endpoint.patch("/policy", description="Patch policy state.", tags=("policy",))
    async def patch_policy(self, input_value, *, context=None):
        return await self.arun(input_value, context=context)

    @endpoint.delete("/policy", tags=("policy",))
    async def delete_policy(self, input_value, *, context=None):
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
        },
        {
            "name": "delete_policy",
            "method": "DELETE",
            "path": "/policy",
            "mode": "run",
            "description": "",
            "tags": ["policy"],
        },
        {
            "name": "patch_policy",
            "method": "PATCH",
            "path": "/policy",
            "mode": "run",
            "description": "Patch policy state.",
            "tags": ["policy"],
        },
    ]
