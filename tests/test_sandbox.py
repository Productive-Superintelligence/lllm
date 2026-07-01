import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from lllm import (
    CallContext,
    SandboxLimitError,
    SandboxPolicy,
    SandboxTimeoutError,
    SandboxedTactic,
    Tactic,
    sandbox_tactic,
)
from lllm.services import create_tactic_app


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        return EchoOutput(text=input_value.text.upper())


class SlowAsyncTactic(Tactic[str, str]):
    name = "slow"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value

    async def _arun(self, input_value, *, context=None):
        await asyncio.sleep(0.05)
        return input_value


@dataclass(frozen=True)
class ResponseSnapshot:
    status_code: int
    content: bytes

    def json(self):
        return json.loads(self.content.decode("utf-8"))


def request(app, method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.request(method, path, **kwargs)
            return ResponseSnapshot(
                status_code=response.status_code,
                content=await response.aread(),
            )

    return asyncio.run(run())


def test_sandboxed_tactic_allows_calls_within_policy():
    tactic = SandboxedTactic(
        EchoTactic(),
        policy=SandboxPolicy(
            max_input_bytes=100,
            max_output_bytes=100,
            allowed_metadata_keys=("tenant",),
        ),
    )

    output = tactic.run(
        {"text": "hello"},
        context=CallContext(metadata={"tenant": "demo"}),
    )

    assert output == EchoOutput(text="HELLO")
    assert tactic.info().runtime_kind == "sandbox"
    assert tactic.info().metadata["sandboxed_tactic"] == "echo"
    assert tactic.info().metadata["sandbox_policy"]["max_input_bytes"] == 100


@pytest.mark.parametrize("name", ["", "   "])
def test_sandboxed_tactic_rejects_explicit_blank_names(name):
    with pytest.raises(ValueError, match="name"):
        SandboxedTactic(EchoTactic(), name=name)


@pytest.mark.parametrize("policy", [[], "", 0, False])
def test_sandboxed_tactic_rejects_falsey_non_mapping_policy(policy):
    with pytest.raises(ValidationError):
        SandboxedTactic(EchoTactic(), policy=policy)


@pytest.mark.parametrize(
    "field",
    ["timeout_seconds", "max_input_bytes", "max_output_bytes"],
)
@pytest.mark.parametrize("value", [False, True])
def test_sandbox_policy_rejects_boolean_numeric_limits(field, value):
    with pytest.raises(ValidationError, match=field):
        SandboxPolicy(**{field: value})

    with pytest.raises(ValidationError, match=field):
        SandboxedTactic(EchoTactic(), policy={field: value})


@pytest.mark.parametrize(
    "allowed_metadata_keys",
    [
        "tenant",
        b"tenant",
        [b"tenant"],
        [""],
        ["   "],
        ["bad key"],
        ["bad/key"],
        ["bad%20key"],
        ["bad;key"],
    ],
)
def test_sandbox_policy_rejects_malformed_allowed_metadata_keys(
    allowed_metadata_keys,
):
    with pytest.raises(ValidationError, match="allowed_metadata_keys"):
        SandboxPolicy(allowed_metadata_keys=allowed_metadata_keys)

    with pytest.raises(ValidationError, match="allowed_metadata_keys"):
        SandboxedTactic(
            EchoTactic(),
            policy={"allowed_metadata_keys": allowed_metadata_keys},
        )


def test_sandbox_policy_accepts_plain_allowed_metadata_keys():
    policy = SandboxPolicy(allowed_metadata_keys=["tenant", "x-api-key", "trace.id"])

    assert policy.allowed_metadata_keys == ("tenant", "x-api-key", "trace.id")


def test_sandboxed_tactic_isolates_policy_object():
    policy = SandboxPolicy(max_input_bytes=100, allowed_metadata_keys=("tenant",))
    tactic = SandboxedTactic(EchoTactic(), policy=policy)

    policy.max_input_bytes = 8
    policy.allowed_metadata_keys = ("tenant", "secret")

    output = tactic.run(
        {"text": "hello"},
        context=CallContext(metadata={"tenant": "demo"}),
    )

    assert output == EchoOutput(text="HELLO")
    assert tactic.policy.max_input_bytes == 100
    assert tactic.policy.allowed_metadata_keys == ("tenant",)


def test_sandboxed_tactic_rejects_input_output_and_metadata_edges():
    small_input = sandbox_tactic(EchoTactic(), {"max_input_bytes": 8})
    small_output = sandbox_tactic(EchoTactic(), {"max_output_bytes": 8})
    metadata_limited = sandbox_tactic(EchoTactic(), {"allowed_metadata_keys": ("tenant",)})

    with pytest.raises(SandboxLimitError, match="input"):
        small_input.run({"text": "hello"})
    with pytest.raises(SandboxLimitError, match="output"):
        small_output.run({"text": "hi"})
    with pytest.raises(SandboxLimitError, match="secret"):
        metadata_limited.run(
            {"text": "hello"},
            context=CallContext(metadata={"secret": "nope"}),
        )


def test_sandboxed_tactic_enforces_async_timeout():
    tactic = SandboxedTactic(
        SlowAsyncTactic(),
        policy=SandboxPolicy(timeout_seconds=0.001),
    )

    async def run():
        await tactic.arun("slow")

    with pytest.raises(SandboxTimeoutError, match="timeout"):
        asyncio.run(run())


def test_sandboxed_tactic_timeout_reaches_fastapi_error_envelope():
    app = create_tactic_app(
        SandboxedTactic(
            SlowAsyncTactic(),
            policy=SandboxPolicy(timeout_seconds=0.001),
        )
    )

    response = request(app, "POST", "/run", json={"input": "slow"})

    assert response.status_code == 500
    detail = response.json()["detail"]["error"]
    assert detail["type"] == "SandboxTimeoutError"
    assert detail["tactic"] == "slow_sandbox"
