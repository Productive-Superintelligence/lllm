"""Small native runtime primitives.

These classes preserve the useful prompt/dialog vocabulary from the original
native runtime without making the public protocol depend on a particular agent
engine or model provider.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import inspect
import re
import string
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictStr,
    field_validator,
)

from ...parsers import (
    BaseParser,
    DefaultTagParser,
    ParseError,
    find_md_blocks,
    find_xml_blocks,
)
from ...protocol._validation import copy_boundary_value, optional_mapping_value


class Role(str, Enum):
    """Message roles used by native dialogs."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    TOOL_CALL = "tool_call"

    @property
    def msg_value(self) -> str:
        if self == Role.SYSTEM:
            return "developer"
        return self.value


Roles = Role


class Modality(str, Enum):
    """Message modality markers."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FUNCTION_CALL = "function_call"


Modalities = Modality


class APIType(str, Enum):
    """Provider API families a native invoker may record."""

    COMPLETION = "completion"
    RESPONSE = "response"


APITypes = APIType


class InvokeCost(BaseModel):
    """Token and cost accounting for a message or dialog."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    audio_prompt_tokens: int = 0
    audio_completion_tokens: int = 0
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cache_read_input_token_cost: float = 0.0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    cost: float = 0.0

    @field_validator(
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "reasoning_tokens",
        "audio_prompt_tokens",
        "audio_completion_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "prompt_cost",
        "completion_cost",
        "cost",
        mode="before",
    )
    @classmethod
    def _reject_bool_numeric_fields(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("usage cost fields must not be boolean")
        return value

    def __add__(self, other: "InvokeCost") -> "InvokeCost":
        return InvokeCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            audio_prompt_tokens=self.audio_prompt_tokens + other.audio_prompt_tokens,
            audio_completion_tokens=self.audio_completion_tokens
            + other.audio_completion_tokens,
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            cost=self.cost + other.cost,
        )


def _usage_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    if not value:
        return 0
    return int(value)


def _usage_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    if not value:
        return 0.0
    return float(value)


def _usage_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


class FunctionCall(BaseModel):
    """One invocation of a native tool, including its result once executed."""

    id: StrictStr = Field(default_factory=lambda: uuid.uuid4().hex)
    name: StrictStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    result_str: StrictStr | None = None
    error_message: StrictStr | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        _validate_tool_name(self.name, "function call name")
        self.arguments = copy_boundary_value(self.arguments)
        self.result = copy_boundary_value(self.result)

    @property
    def success(self) -> bool:
        return self.error_message is None and self.result_str is not None

    def equals(self, other: "FunctionCall") -> bool:
        return self.name == other.name and self.arguments == other.arguments

    def is_repeated(self, function_calls: list["FunctionCall"]) -> bool:
        return any(self.equals(function_call) for function_call in function_calls)

    def __str__(self) -> str:
        result = f"Calling function: {self.name} with arguments: {self.arguments}\n"
        if self.success:
            result += f"Return:\n---\n{self.result_str}\n---\n"
        if self.error_message:
            result += f"Error: {self.error_message}\n"
        return result


class TokenLogprob(BaseModel):
    """Provider logprob data attached to a token."""

    token: StrictStr | None = None
    logprob: float | None = None
    bytes: list[int] | None = None
    top_logprobs: list["TokenLogprob"] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    @field_validator("logprob", mode="before")
    @classmethod
    def _reject_bool_logprob(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("logprob must not be boolean")
        return value

    @field_validator("bytes", mode="before")
    @classmethod
    def _reject_bool_bytes(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, bool) for item in value
        ):
            raise ValueError("bytes must contain integers, not booleans")
        return value

    def model_post_init(self, __context: Any) -> None:
        self.bytes = copy.deepcopy(self.bytes)
        self.top_logprobs = copy.deepcopy(self.top_logprobs)


class Message(BaseModel):
    """A native dialog message with provider-neutral metadata."""

    role: Role
    content: StrictStr | list[dict[str, Any]]
    name: StrictStr
    function_calls: list[FunctionCall] = Field(default_factory=list)
    modality: Modality = Modality.TEXT
    logprobs: list[TokenLogprob] = Field(default_factory=list)
    parsed: dict[str, Any] = Field(default_factory=dict)
    model: StrictStr | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    api_type: APIType = APIType.COMPLETION
    vectors: list[float] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.content = copy_boundary_value(self.content)
        self.function_calls = copy.deepcopy(self.function_calls)
        self.logprobs = copy.deepcopy(self.logprobs)
        self.parsed = copy_boundary_value(self.parsed)
        self.usage = copy_boundary_value(self.usage)
        self.metadata = copy_boundary_value(self.metadata)
        self.vectors = copy.deepcopy(self.vectors)

    @property
    def sanitized_name(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", self.name)[:64]

    @property
    def is_function_call(self) -> bool:
        return bool(self.function_calls)

    @property
    def cost(self) -> InvokeCost:
        if not self.usage:
            return InvokeCost()
        prompt_tokens = _usage_int(
            self.usage.get("prompt_tokens"),
            "usage.prompt_tokens",
        )
        completion_tokens = _usage_int(
            self.usage.get("completion_tokens"),
            "usage.completion_tokens",
        )
        prompt_details = _usage_mapping(
            self.usage.get("prompt_tokens_details"),
            "usage.prompt_tokens_details",
        )
        completion_details = _usage_mapping(
            self.usage.get("completion_tokens_details"),
            "usage.completion_tokens_details",
        )
        return InvokeCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=_usage_int(
                self.usage.get("total_tokens", prompt_tokens + completion_tokens),
                "usage.total_tokens",
            ),
            cached_prompt_tokens=_usage_int(
                prompt_details.get("cached_tokens"),
                "usage.prompt_tokens_details.cached_tokens",
            ),
            audio_prompt_tokens=_usage_int(
                prompt_details.get("audio_tokens"),
                "usage.prompt_tokens_details.audio_tokens",
            ),
            reasoning_tokens=_usage_int(
                completion_details.get("reasoning_tokens"),
                "usage.completion_tokens_details.reasoning_tokens",
            ),
            audio_completion_tokens=_usage_int(
                completion_details.get("audio_tokens"),
                "usage.completion_tokens_details.audio_tokens",
            ),
            input_cost_per_token=_usage_float(
                self.usage.get("input_cost_per_token"),
                "usage.input_cost_per_token",
            ),
            output_cost_per_token=_usage_float(
                self.usage.get("output_cost_per_token"),
                "usage.output_cost_per_token",
            ),
            cache_read_input_token_cost=_usage_float(
                self.usage.get("cache_read_input_token_cost"),
                "usage.cache_read_input_token_cost",
            ),
            prompt_cost=_usage_float(
                self.usage.get("prompt_cost"),
                "usage.prompt_cost",
            ),
            completion_cost=_usage_float(
                self.usage.get("completion_cost"),
                "usage.completion_cost",
            ),
            cost=_usage_float(
                self.usage.get("response_cost"),
                "usage.response_cost",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls.model_validate(data)


@dataclass
class DialogTreeNode:
    """Serializable lineage metadata for a dialog branch."""

    dialog_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    owner: str | None = None
    parent_id: str | None = None
    split_point: int | None = None
    children_ids: list[str] = field(default_factory=list)
    last_n: int | None = None
    first_k: int | None = None
    _parent: Optional["DialogTreeNode"] = field(default=None, repr=False)
    _children: list["DialogTreeNode"] = field(default_factory=list, repr=False)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def depth(self) -> int:
        depth = 0
        node = self
        while node._parent is not None:
            depth += 1
            node = node._parent
        return depth

    def add_child(self, child: "DialogTreeNode") -> None:
        child._parent = self
        child.parent_id = self.dialog_id
        self._children.append(child)
        if child.dialog_id not in self.children_ids:
            self.children_ids.append(child.dialog_id)

    def subtree_ids(self) -> list[str]:
        visited: list[str] = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            visited.append(node.dialog_id)
            queue.extend(node._children)
        return visited

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialog_id": self.dialog_id,
            "owner": self.owner,
            "parent_id": self.parent_id,
            "split_point": self.split_point,
            "children_ids": list(self.children_ids),
            "last_n": self.last_n,
            "first_k": self.first_k,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DialogTreeNode":
        return cls(
            dialog_id=data["dialog_id"],
            owner=data.get("owner"),
            parent_id=data.get("parent_id"),
            split_point=data.get("split_point"),
            children_ids=list(data.get("children_ids", [])),
            last_n=data.get("last_n"),
            first_k=data.get("first_k"),
        )


@dataclass
class Dialog:
    """Append-only native message history with fork lineage."""

    session_name: str | None = None
    top_prompt: Optional["Prompt"] = None
    owner: str | None = None
    _messages: list[Message] = field(default_factory=list)
    tree_node: DialogTreeNode | None = None
    _parent_dialog: Optional["Dialog"] = field(default=None, repr=False)
    _children_dialogs: list["Dialog"] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.tree_node is None:
            self.tree_node = DialogTreeNode(owner=self.owner)
        if self.session_name is None:
            timestamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.session_name = f"{timestamp}_{uuid.uuid4().hex[:6]}"

    @property
    def dialog_id(self) -> str:
        assert self.tree_node is not None
        return self.tree_node.dialog_id

    @property
    def parent(self) -> Optional["Dialog"]:
        return self._parent_dialog

    @property
    def children(self) -> list["Dialog"]:
        return list(self._children_dialogs)

    @property
    def is_root(self) -> bool:
        assert self.tree_node is not None
        return self.tree_node.is_root

    @property
    def depth(self) -> int:
        assert self.tree_node is not None
        return self.tree_node.depth

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def head(self) -> Message | None:
        return self._messages[0] if self._messages else None

    @property
    def tail(self) -> Message | None:
        return self._messages[-1] if self._messages else None

    @property
    def cost(self) -> InvokeCost:
        total = InvokeCost()
        for message in self._messages:
            total += message.cost
        return total

    def append(self, message: Message) -> None:
        message.metadata["dialog_id"] = self.dialog_id
        self._messages.append(message)

    def put_text(
        self,
        text: str,
        *,
        name: str = "user",
        role: Role = Role.USER,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            role=role,
            content=text,
            name=name,
            metadata=_metadata_mapping(metadata),
        )
        self.append(message)
        return message

    def put_prompt(
        self,
        prompt: "Prompt",
        *,
        prompt_args: dict[str, Any] | None = None,
        name: str = "user",
        role: Role = Role.USER,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        content = prompt(**_optional_mapping("prompt_args", prompt_args))
        message = Message(
            role=role,
            content=content,
            name=name,
            metadata=_metadata_mapping(metadata),
        )
        self.append(message)
        self.top_prompt = prompt
        return message

    def fork(self, *, last_n: int = 0, first_k: int = 1) -> "Dialog":
        assert self.tree_node is not None
        if last_n >= len(self._messages):
            last_n = 0
        if last_n > 0:
            tail_start = len(self._messages) - last_n
            first_k = min(first_k, tail_start) if first_k > 0 else 0
            messages = self._messages[:first_k] + self._messages[tail_start:]
        else:
            messages = self._messages
        child_node = DialogTreeNode(
            owner=self.owner,
            split_point=len(messages),
            last_n=last_n,
            first_k=first_k,
        )
        child = Dialog(
            session_name=self.session_name,
            top_prompt=self.top_prompt,
            owner=self.owner,
            _messages=[copy.deepcopy(message) for message in messages],
            tree_node=child_node,
        )
        self.tree_node.add_child(child_node)
        child._parent_dialog = self
        self._children_dialogs.append(child)
        return child

    def overview(self, *, max_length: int = 100, remove_tail: bool = False) -> str:
        remove_tail_value = _bool_value("remove_tail", remove_tail)
        messages = self._messages[:-1] if remove_tail_value else self._messages
        rows = []
        for index, message in enumerate(messages):
            content = str(message.content)
            preview = content[:max_length] + "..." if len(content) > max_length else content
            rows.append(f"[{index}. {message.name} ({message.role.msg_value})]: {preview}")
        return "\n\n".join(rows)

    def tree_overview(self, *, indent: int = 0) -> str:
        assert self.tree_node is not None
        prefix = "  " * indent
        branch = "+- " if indent > 0 else ""
        line = (
            f"{prefix}{branch}[{self.dialog_id[:8]}] owner={self.owner} "
            f"msgs={len(self._messages)} split@{self.tree_node.split_point}"
        )
        lines = [line]
        for child in self._children_dialogs:
            lines.append(child.tree_overview(indent=indent + 1))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        assert self.tree_node is not None
        return {
            "messages": [message.to_dict() for message in self._messages],
            "session_name": self.session_name,
            "owner": self.owner,
            "tree_node": self.tree_node.to_dict(),
            "top_prompt": self.top_prompt.to_dict() if self.top_prompt else None,
            "children": [child.to_dict() for child in self._children_dialogs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dialog":
        top_prompt_data = data.get("top_prompt")
        dialog = cls(
            _messages=[Message.from_dict(message) for message in data.get("messages", [])],
            session_name=data.get("session_name"),
            owner=data.get("owner"),
            tree_node=DialogTreeNode.from_dict(data["tree_node"])
            if data.get("tree_node")
            else None,
            top_prompt=Prompt.from_dict(top_prompt_data) if top_prompt_data else None,
        )
        assert dialog.tree_node is not None
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            assert child.tree_node is not None
            child._parent_dialog = dialog
            dialog.tree_node.add_child(child.tree_node)
            dialog._children_dialogs.append(child)
        return dialog


_PY_TYPE_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _default_function_call_processor(result: Any, function_call: FunctionCall) -> str:
    return (
        f"Return of calling function {function_call.name} "
        f"with arguments {function_call.arguments}:\n---\n{result}\n---\n"
    )


def _json_type_for_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _json_type_for_annotation(args[0]) if args else "string"
    if origin is not None:
        return _PY_TYPE_TO_JSON.get(origin, "string")
    return _PY_TYPE_TO_JSON.get(annotation, "string")


class Function(BaseModel):
    """Declarative native tool schema with an optional Python implementation."""

    name: StrictStr
    description: StrictStr
    properties: dict[str, Any]
    required: list[StrictStr] = Field(default_factory=list)
    additional_properties: StrictBool = False
    strict: StrictBool = True
    function: Callable[..., Any] | None = None
    processor: Callable[[Any, FunctionCall], str] = _default_function_call_processor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        _validate_tool_name(self.name, "function name")
        self.properties = copy_boundary_value(self.properties)
        self.required = copy.deepcopy(self.required)

    def __call__(self, function_call: FunctionCall) -> FunctionCall:
        if self.function is None:
            raise RuntimeError(f"Function '{self.name}' has no implementation.")
        try:
            result = self.function(**function_call.arguments)
        except Exception as exc:  # pragma: no cover - exact user function varies
            function_call.error_message = str(exc)
            function_call.result_str = f"Error: {exc}"
            return function_call
        function_call.result = result
        function_call.result_str = self.processor(result, function_call)
        return function_call

    def to_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": copy_boundary_value(self.properties),
                    "required": copy.deepcopy(self.required),
                    "additionalProperties": self.additional_properties,
                },
                "strict": self.strict,
            },
        }

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        prop_desc: Mapping[str, str] | None = None,
        strict: bool = True,
        processor: Callable[[Any, FunctionCall], str] = _default_function_call_processor,
    ) -> "Function":
        function_name = name if name is not None else fn.__name__
        signature = inspect.signature(fn)
        hints = get_type_hints(fn) if getattr(fn, "__annotations__", None) else {}
        property_descriptions = _optional_string_mapping("prop_desc", prop_desc)
        strict_value = _bool_value("strict", strict)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, parameter in signature.parameters.items():
            if param_name in {"self", "cls"}:
                continue
            annotation = hints.get(param_name, str)
            schema: dict[str, Any] = {"type": _json_type_for_annotation(annotation)}
            if param_name in property_descriptions:
                schema["description"] = property_descriptions[param_name]
            if parameter.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                default_note = f"(default: {parameter.default!r})"
                if schema.get("description"):
                    schema["description"] = f"{schema['description']} {default_note}"
                else:
                    schema["description"] = default_note
            properties[param_name] = schema
        return cls(
            name=function_name,
            description=description or inspect.getdoc(fn) or function_name,
            properties=properties,
            required=required,
            strict=strict_value,
            function=fn,
            processor=processor,
        )


def _validate_tool_name(value: str, label: str) -> None:
    if (
        not value.strip()
        or value in {".", ".."}
        or "%" in value
        or any(ch.isspace() for ch in value)
        or any(ch in value for ch in "/:\\")
    ):
        raise ValueError(
            f"{label} must be a non-empty token without whitespace, "
            "percent escapes, or path separators."
        )


def tool(
    description: str | None = None,
    prop_desc: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    strict: bool = True,
    processor: Callable[[Any, FunctionCall], str] = _default_function_call_processor,
) -> Callable[[Callable[..., Any]], Function]:
    """Decorate a plain Python callable as a native :class:`Function`."""

    strict_value = _bool_value("strict", strict)

    def decorator(fn: Callable[..., Any]) -> Function:
        return Function.from_callable(
            fn,
            name=name,
            description=description,
            prop_desc=prop_desc,
            strict=strict_value,
            processor=processor,
        )

    return decorator


class StringFormatterRenderer(BaseModel):
    """Render prompt templates with ``str.format``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def render(self, template: str, **kwargs: Any) -> str:
        return template.format(**kwargs)


class Prompt(BaseModel):
    """A native prompt template plus lightweight parser/tool metadata."""

    path: StrictStr
    prompt: StrictStr
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser: Any = None
    format: Any = None
    function_list: list[Function | StrictStr] = Field(default_factory=list)
    addon_args: dict[str, Any] = Field(default_factory=dict)
    renderer: Any = Field(default_factory=StringFormatterRenderer)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _template_vars: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        formatter = string.Formatter()
        self._template_vars = {
            field_name.split(".")[0].split("[")[0]
            for _, field_name, _, _ in formatter.parse(self.prompt)
            if field_name is not None
        }
        self.metadata = copy_boundary_value(self.metadata)
        self.addon_args = copy_boundary_value(self.addon_args)
        self.function_list = copy.deepcopy(self.function_list)

    @property
    def functions(self) -> dict[str, Function]:
        return {
            function.name: copy.deepcopy(function)
            for function in self.function_list
            if isinstance(function, Function)
        }

    @property
    def template_vars(self) -> set[str]:
        return set(self._template_vars)

    def validate_args(self, prompt_args: dict[str, Any]) -> list[str]:
        return sorted(var for var in self._template_vars if var not in prompt_args)

    def __call__(self, **kwargs: Any) -> str:
        missing = self.validate_args(kwargs)
        if missing:
            raise ValueError(
                f"Missing required template variables for prompt '{self.path}': {missing}"
            )
        return self.renderer.render(self.prompt, **kwargs)

    def parse(self, content: str, **runtime_args: Any) -> dict[str, Any]:
        if self.parser is None:
            return {"raw": content}
        if _has_static_callable(self.parser, "parse"):
            parsed = self.parser.parse(content, **runtime_args)
        elif callable(self.parser):
            parsed = self.parser(content, **runtime_args)
        else:
            raise TypeError("parser must be callable or expose a callable parse() method.")
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        parsed.setdefault("raw", content)
        return parsed

    def get_function(self, name: str) -> Function:
        try:
            return self.functions[name]
        except KeyError as exc:
            raise KeyError(
                f"Function '{name}' not found on prompt '{self.path}'. "
                f"Available: {sorted(self.functions)}"
            ) from exc

    def extend(self, **overrides: Any) -> "Prompt":
        if "path" not in overrides:
            raise ValueError("extend() requires a new 'path'")
        current = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if field_name in {"addon_args", "metadata"}:
                value = copy_boundary_value(value)
            elif field_name == "function_list":
                value = copy.deepcopy(value)
            current[field_name] = value
        current.update(overrides)
        return Prompt(**current)

    def info_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "prompt_hash": hashlib.sha256(self.prompt.encode()).hexdigest()[:12],
            "metadata": copy_boundary_value(self.metadata),
            "functions": [
                function.name if isinstance(function, Function) else function
                for function in self.function_list
            ],
            "addon_args": copy_boundary_value(self.addon_args),
            "has_parser": self.parser is not None,
            "has_format": self.format is not None,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"parser", "renderer"})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        return cls.model_validate(data)


def _metadata_mapping(value: Any) -> dict[str, Any]:
    return _optional_mapping("metadata", value)


def _optional_mapping(label: str, value: Any) -> dict[str, Any]:
    return optional_mapping_value(label, value)


def _optional_string_mapping(label: str, value: Any) -> dict[str, str]:
    entries = _optional_mapping(label, value)
    if not all(
        isinstance(entry_key, str) and isinstance(entry_value, str)
        for entry_key, entry_value in entries.items()
    ):
        raise TypeError(f"{label} keys and values must be strings.")
    return entries


def _bool_value(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _has_static_callable(value: Any, name: str) -> bool:
    try:
        attribute = inspect.getattr_static(value, name)
    except AttributeError:
        return False
    if isinstance(attribute, (classmethod, staticmethod)):
        return callable(attribute.__func__)
    return callable(attribute)


__all__ = [
    "APITypes",
    "APIType",
    "BaseParser",
    "DefaultTagParser",
    "Dialog",
    "DialogTreeNode",
    "Function",
    "FunctionCall",
    "InvokeCost",
    "Message",
    "Modalities",
    "Modality",
    "ParseError",
    "Prompt",
    "Role",
    "Roles",
    "StringFormatterRenderer",
    "TokenLogprob",
    "find_md_blocks",
    "find_xml_blocks",
    "tool",
]
