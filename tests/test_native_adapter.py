import pytest
from pydantic import BaseModel

from lllm.runtimes.native import FunctionCall, NativeTacticAdapter, tactic_as_function
from lllm.runtimes.python import as_tactic


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


@pytest.mark.parametrize("name", ["", "   "])
def test_native_adapter_rejects_explicit_blank_names(name):
    with pytest.raises(ValueError, match="name"):
        NativeTacticAdapter(NativeAdd(), name=name)


def test_protocol_tactic_can_be_native_function_tool():
    def add(task: AddInput) -> int:
        return task.left + task.right

    tactic = as_tactic(add, name="adder")
    function = tactic_as_function(tactic, parameter_mode="kwargs")
    call = function(FunctionCall(name=function.name, arguments={"left": 2, "right": 4}))

    assert function.name == "adder"
    assert function.required == ["left", "right"]
    assert call.success
    assert call.result == 6
