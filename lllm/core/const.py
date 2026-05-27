from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from .dialog import Message


@dataclass
class InvokeResult:
    """
    Per-invocation diagnostics and the message returned by an invoker.

    Attributes:
        raw_response:    The raw API response object (completion, response, etc.).
        model_args:      The actual model args sent to the API (after merging).
        execution_errors: Parse/validation errors encountered during this invocation.
        message: The message object returned by the invoker.
    """

    raw_response: Any = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    execution_errors: List[Exception] = field(default_factory=list)
    message: Optional[Message] = (
        None  # always set by invoker, None is just the dataclass default
    )

    @property
    def has_errors(self) -> bool:
        return len(self.execution_errors) > 0

    @property
    def cost(self) -> InvokeCost:
        return self.message.cost if self.message else InvokeCost()

    @property
    def error_message(self) -> str:
        return "\n".join(str(e) for e in self.execution_errors)


class FunctionCall(BaseModel):
    """One invocation of a tool, including its result once executed."""

    id: str
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    result_str: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def success(self):
        return self.error_message is None and self.result_str is not None

    def __str__(self):
        _str = f"Calling function: {self.name} with arguments: {self.arguments}\n"
        if self.success:
            _str += f"Return:\n---\n{self.result_str}\n---\n"
        return _str

    def equals(self, other: FunctionCall) -> bool:
        if self.name != other.name:
            return False
        if set(self.arguments.keys()) != set(other.arguments.keys()):
            return False
        for k, v in self.arguments.items():
            if other.arguments[k] != v:
                return False
        return True

    def is_repeated(self, function_calls: List[FunctionCall]) -> bool:
        return any(self.equals(fc) for fc in function_calls)


class ParseError(Exception):
    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class Roles(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"  # only use tool role, not function role
    TOOL_CALL = "tool_call"

    @property
    def msg_value(self):
        if self == Roles.SYSTEM:
            return "developer"
        return self.value


class Invokers(str, Enum):
    LITELLM = "litellm"


class Modalities(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FUNCTION_CALL = "function_call"


class APITypes(str, Enum):
    COMPLETION = "completion"
    RESPONSE = "response"


LLM_SIDE_ROLES = [Roles.ASSISTANT, Roles.TOOL_CALL]


class InvokeCost(BaseModel):
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    audio_prompt_tokens: int = 0
    audio_completion_tokens: int = 0

    # Rates (USD per token, recorded at invocation time)
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cache_read_input_token_cost: float = 0.0

    # Calculated Dollar Costs
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    cost: float = 0.0  # Total combined cost

    def __str__(self):
        rates = []
        if self.input_cost_per_token:
            rates.append(f"In: ${self.input_cost_per_token:.7f}/tok")
        if self.output_cost_per_token:
            rates.append(f"Out: ${self.output_cost_per_token:.7f}/tok")
        rates_str = f" | Rates: [{', '.join(rates)}]" if rates else ""

        return (
            f"Tokens: {self.total_tokens} (Prompt: {self.prompt_tokens} "
            f"[Cached: {self.cached_prompt_tokens}], "
            f"Completion: {self.completion_tokens} "
            f"[Reasoning: {self.reasoning_tokens}]){rates_str}\n"
            f"Cost Breakdown: Prompt: ${self.prompt_cost:.6f}, Completion: ${self.completion_cost:.6f} | "
            f"Total Cost: ${self.cost:.6f}"
        )

    def __add__(self, other: "InvokeCost") -> "InvokeCost":
        return InvokeCost(
            # Token counts — additive
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            audio_prompt_tokens=self.audio_prompt_tokens + other.audio_prompt_tokens,
            audio_completion_tokens=self.audio_completion_tokens
            + other.audio_completion_tokens,
            # Rates — NOT additive, zero them out in aggregates
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            cache_read_input_token_cost=0.0,
            # Dollar costs — additive
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            cost=self.cost + other.cost,
        )
