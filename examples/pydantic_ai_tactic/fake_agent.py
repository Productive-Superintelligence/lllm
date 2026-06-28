from lllm.runtimes import PydanticAITactic


class Result:
    def __init__(self, output):
        self.output = output


class FakeAgent:
    name = "fake-agent"
    output_type = str

    def run_sync(self, task, **kwargs):
        return Result(task.upper())


def build_tactic():
    return PydanticAITactic(FakeAgent(), input_type=str, output_type=str)
