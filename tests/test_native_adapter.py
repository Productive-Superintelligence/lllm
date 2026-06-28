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
    tactic = NativeTacticAdapter(NativeAdd())

    assert tactic.info().runtime_kind == "native"
    assert tactic.run({"left": 2, "right": 3}) == 5
