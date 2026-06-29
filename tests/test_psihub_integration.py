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
    example = {"input": {"text": "forward"}, "output": {"text": "forward"}}
    resource = tactic_resource(PolicyTactic(examples=[example]))

    assert resource["name"] == "policy"
    assert resource["input_schema"]["properties"]["text"]["type"] == "string"
    assert resource["output_schema"]["properties"]["text"]["type"] == "string"
    assert resource["examples"] == [example]
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


def test_tactic_resource_isolates_exported_mutable_metadata():
    example = {"input": {"items": ["one"]}}
    metadata = {"labels": ["policy"]}
    info = PolicyTactic(examples=[example], metadata=metadata).info()

    class CachedInfoTactic(PolicyTactic):
        def info(self):
            return info

    resource = tactic_resource(CachedInfoTactic())

    resource["input_schema"]["properties"]["text"]["type"] = "integer"
    resource["output_schema"]["properties"]["text"]["type"] = "integer"
    resource["examples"][0]["input"]["items"].append("changed")
    resource["metadata"]["labels"].append("changed")

    assert info.input_schema["properties"]["text"]["type"] == "string"
    assert info.output_schema["properties"]["text"]["type"] == "string"
    assert info.examples == [{"input": {"items": ["one"]}}]
    assert info.metadata == {"labels": ["policy"]}
