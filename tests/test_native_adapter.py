import pytest
from pydantic import BaseModel

from lllm import CallContext
from lllm.runtimes.native import FunctionCall, NativeTacticAdapter, Tactic, tactic_as_function
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


class NativeContextEcho(Tactic):
    name = "context-echo"
    input_type = str
    output_type = str

    def call(self, task, *, context=None, suffix=""):
        trace_id = context.trace_id if context is not None else "missing"
        return f"{task}:{trace_id}{suffix}"


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


def test_native_adapter_forwards_context_to_native_tactic_call_when_supported():
    tactic = NativeTacticAdapter(NativeContextEcho(), run_kwargs={"suffix": "!"})

    output = tactic.run("hello", context=CallContext(trace_id="trace-native"))

    assert output == "hello:trace-native!"


def test_native_adapter_does_not_force_context_into_native_tactic_call():
    tactic = NativeTacticAdapter(NativeAdd())

    assert tactic.run({"left": 2, "right": 3}, context=CallContext()) == 5


def test_native_adapter_passes_runtime_kwargs_and_context_metadata_to_plain_object():
    class NativeRunner:
        name = "runner"
        input_type = str
        output_type = dict

        def __init__(self):
            self.seen = None

        def run(self, task, *, metadata=None, tone="plain"):
            self.seen = {"task": task, "metadata": metadata, "tone": tone}
            return self.seen

    native = NativeRunner()
    tactic = NativeTacticAdapter(native, run_kwargs={"tone": "direct"})
    context = CallContext(
        trace_id="trace-meta",
        metadata={"caller": "test"},
        tags={"kind": "native"},
    )

    output = tactic.run("brief", context=context)

    assert output["tone"] == "direct"
    assert output["metadata"]["caller"] == "test"
    assert output["metadata"]["lllm_trace_id"] == "trace-meta"
    assert output["metadata"]["lllm_tags"] == {"kind": "native"}
    assert native.seen == output


def test_native_adapter_bridges_native_stream():
    class NativeStreamer:
        name = "native-streamer"
        input_type = str
        output_type = str

        def run(self, task):
            return task

        def stream(self, task, *, suffix=""):
            yield task
            yield f"{task.upper()}{suffix}"

    tactic = NativeTacticAdapter(NativeStreamer(), run_kwargs={"suffix": "!"})

    assert tactic.capabilities() == {"run", "arun", "stream"}
    assert list(tactic.stream("hi")) == ["hi", "HI!"]


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
