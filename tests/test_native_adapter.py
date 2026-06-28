from pydantic import BaseModel

from lllm.runtimes.native import NativeTacticAdapter


class AddInput(BaseModel):
    left: int
    right: int


class NativeAdd:
    name = "add"
    input_model = AddInput
    output_type = int

    def call(self, task):
        return task.left + task.right


def test_native_adapter_keeps_native_object_behind_boundary():
    example = {"input": {"left": 2, "right": 3}, "output": 5}
    tactic = NativeTacticAdapter(
        NativeAdd(),
        description="Native addition.",
        package_ref="psi://demo/native/tactics/add",
        service_ref="psi://demo/native/services/api",
        examples=[example],
        metadata={"owner": "tests"},
    )
    info = tactic.info()

    assert info.runtime_kind == "native"
    assert info.description == "Native addition."
    assert info.package_ref == "psi://demo/native/tactics/add"
    assert info.service_ref == "psi://demo/native/services/api"
    assert info.examples == [example]
    assert info.metadata == {"owner": "tests"}
    assert tactic.run({"left": 2, "right": 3}) == 5
