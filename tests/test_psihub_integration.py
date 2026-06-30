from types import MappingProxyType, SimpleNamespace

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


def test_tactic_resource_filters_secret_examples_and_metadata():
    resource = tactic_resource(
        PolicyTactic(
            examples=[
                {
                    "input": {
                        "text": "forward",
                        "headers": {
                            "authorization": "Bearer raw-example-auth",
                            "x-api-key": "raw-example-key",
                            "x-policy": "safe-policy",
                        },
                    },
                    "output": {
                        "password": "raw-example-password",
                        "text": "forward",
                    },
                }
            ],
            metadata={
                "api_key_ref": "credentials/openai",
                "headers": {
                    "authorization": "Bearer raw-metadata-auth",
                    "x-policy": "safe-metadata-policy",
                },
            },
        )
    )

    rendered = str(resource["examples"]) + str(resource["metadata"])
    assert "raw-example-auth" not in rendered
    assert "raw-example-key" not in rendered
    assert "raw-example-password" not in rendered
    assert "raw-metadata-auth" not in rendered
    assert "safe-policy" in rendered
    assert "safe-metadata-policy" in rendered
    assert resource["metadata"]["api_key_ref"] == "credentials/openai"


def test_tactic_resource_accepts_nested_read_only_mapping_info_values():
    input_inner = {"type": "string"}
    output_inner = {"type": "string"}
    example_inner = {"text": "forward"}
    metadata_inner = {"labels": ["policy"]}

    class CustomInfoTactic(PolicyTactic):
        def info(self):
            return SimpleNamespace(
                name="policy",
                description="",
                runtime_kind="python",
                capabilities=("run",),
                input_schema={"properties": {"text": MappingProxyType(input_inner)}},
                output_schema={"properties": {"text": MappingProxyType(output_inner)}},
                package_ref=None,
                service_ref=None,
                examples=({"input": MappingProxyType(example_inner)},),
                metadata={"nested": MappingProxyType(metadata_inner)},
            )

    resource = tactic_resource(CustomInfoTactic())
    input_inner["type"] = "integer"
    output_inner["type"] = "integer"
    example_inner["text"] = "changed"
    metadata_inner["labels"].append("changed")

    assert resource["input_schema"] == {"properties": {"text": {"type": "string"}}}
    assert resource["output_schema"] == {"properties": {"text": {"type": "string"}}}
    assert resource["examples"] == ({"input": {"text": "forward"}},)
    assert resource["metadata"] == {"nested": {"labels": ["policy"]}}
