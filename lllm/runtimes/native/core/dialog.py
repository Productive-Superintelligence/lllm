from __future__ import annotations

import base64
import copy
import datetime as dt
import io
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from ....protocol._validation import (
    copy_boundary_value,
    mapping_field_value,
    mapping_sequence_field_value,
    metadata_field_value,
    optional_mapping_value,
    optional_metadata_mapping_value,
)
from .. import utils as U
from .const import APITypes, FunctionCall, InvokeCost, Modalities, Roles
from .runtime import Runtime, get_default_runtime

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .prompt import Prompt


def _is_pil_image(obj: Any) -> bool:
    """Return True if *obj* is a PIL/Pillow Image without requiring PIL at import time."""
    try:
        from PIL import Image

        return isinstance(obj, Image.Image)
    except ImportError:
        return False


def _media_type(data: bytes) -> Optional[str]:
    """The media type of PNG, JPEG, WebP, or GIF data, from its first bytes."""
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_media_type(image_base64: str) -> Optional[str]:
    """The media type of base64 image data, decoding only its first bytes."""
    head = "".join(image_base64[:32].split())
    try:
        return _media_type(base64.b64decode(head[: len(head) // 4 * 4]))
    except ValueError:  # not base64
        return None


def _image_parts(message: "Message") -> List[Dict[str, Any]]:
    """An image message as chat content parts: its caption, then the image."""
    parts: List[Dict[str, Any]] = []
    if "caption" in message.metadata:
        parts.append({"type": "text", "text": message.metadata["caption"]})
    media_type = _image_media_type(message.content) or "image/jpeg"
    url = f"data:{media_type};base64,{message.content}"
    parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _image_placeholder(message: "Message") -> str:
    """``<image/png, 2.0 KB> caption``: an image message without its data."""
    content = message.content if isinstance(message.content, str) else ""
    data = "".join(content.split())
    size = len(data) * 3 // 4 - data[-2:].count("=")
    if size < 1024:
        shown = f"{size} B"
    elif size < 1024 * 1024:
        shown = f"{size / 1024:.1f} KB"
    else:
        shown = f"{size / (1024 * 1024):.1f} MB"
    placeholder = f"<{_image_media_type(content) or 'image'}, {shown}>"
    caption = message.metadata.get("caption")
    return f"{placeholder} {caption}" if caption else placeholder


class TokenLogprob(BaseModel):
    token: Optional[StrictStr] = None
    logprob: Optional[float] = None
    bytes: Optional[List[int]] = None
    top_logprobs: List["TokenLogprob"] = Field(default_factory=list)

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


TokenLogprob.model_rebuild()

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def _sanitize_name(raw_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)[:64]


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


def _usage_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not value:
        return {}
    if not isinstance(value, dict):
        if hasattr(value, "items"):
            return dict(value)
        raise ValueError(f"{label} must be a mapping")
    return value


def _metadata_mapping(value: Any) -> Dict[str, Any]:
    return optional_metadata_mapping_value("metadata", value)


def _optional_mapping(label: str, value: Any) -> Dict[str, Any]:
    return optional_mapping_value(label, value)


def _bool_value(label: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _nonnegative_int_value(label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


class Message(BaseModel):
    role: Roles
    content: Union[
        StrictStr, List[Dict[str, Any]]
    ]  # Content can be string or list of content parts (for images)
    name: StrictStr  # name of the sender
    function_calls: List[FunctionCall] = Field(default_factory=list)
    modality: Modalities = Modalities.TEXT
    logprobs: List[TokenLogprob] = Field(default_factory=list)
    parsed: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[StrictStr] = None
    usage: Dict[str, Any] = Field(default_factory=dict)  #
    metadata: Dict[str, Any] = Field(default_factory=dict)
    api_type: APITypes = APITypes.COMPLETION

    vectors: List[float] = Field(
        default_factory=list
    )  # place holder for embedding vectors of the message, can be used for training, similarity search, etc. Need special invoker to support this.
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
    def sanitized_name(self):
        return _sanitize_name(self.name)

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return mapping_sequence_field_value("content", value)
        return value

    @field_validator("parsed", "usage", mode="before")
    @classmethod
    def _validate_public_maps(cls, value: Any, info: Any) -> Any:
        return mapping_field_value(info.field_name, value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return metadata_field_value("metadata", value)

    @field_validator("logprobs", mode="before")
    @classmethod
    def _coerce_logprobs(cls, value):
        if not value:
            return []
        normalized: List[TokenLogprob] = []
        for entry in value:
            if isinstance(entry, TokenLogprob):
                normalized.append(entry)
                continue
            if isinstance(entry, dict):
                normalized.append(TokenLogprob(**entry))
                continue
            if isinstance(entry, (int, float)):
                normalized.append(TokenLogprob(logprob=float(entry)))
                continue
            normalized.append(TokenLogprob(token=str(entry)))
        return normalized

    @property
    def cost(self) -> InvokeCost:
        if not self.usage:
            return InvokeCost()

        p_tokens = _usage_int(self.usage.get("prompt_tokens"), "usage.prompt_tokens")
        c_tokens = _usage_int(
            self.usage.get("completion_tokens"), "usage.completion_tokens"
        )
        t_tokens = _usage_int(
            self.usage.get("total_tokens", p_tokens + c_tokens),
            "usage.total_tokens",
        )
        p_details = _usage_mapping(
            self.usage.get("prompt_tokens_details"), "usage.prompt_tokens_details"
        )
        c_details = _usage_mapping(
            self.usage.get("completion_tokens_details"),
            "usage.completion_tokens_details",
        )

        return InvokeCost(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            cached_prompt_tokens=_usage_int(
                p_details.get("cached_tokens"),
                "usage.prompt_tokens_details.cached_tokens",
            ),
            audio_prompt_tokens=_usage_int(
                p_details.get("audio_tokens"),
                "usage.prompt_tokens_details.audio_tokens",
            ),
            reasoning_tokens=_usage_int(
                c_details.get("reasoning_tokens"),
                "usage.completion_tokens_details.reasoning_tokens",
            ),
            audio_completion_tokens=_usage_int(
                c_details.get("audio_tokens"),
                "usage.completion_tokens_details.audio_tokens",
            ),
            # Dollar costs
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
            prompt_cost=_usage_float(self.usage.get("prompt_cost"), "usage.prompt_cost"),
            completion_cost=_usage_float(
                self.usage.get("completion_cost"), "usage.completion_cost"
            ),
            cost=_usage_float(self.usage.get("response_cost"), "usage.response_cost"),
        )

    @property
    def is_function_call(self) -> bool:
        return len(self.function_calls) > 0

    def to_dict(self):
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict):
        return cls.model_validate(d)


Message.model_rebuild()


# ---------------------------------------------------------------------------
# DialogTree and Dialog
# ---------------------------------------------------------------------------


@dataclass
class DialogTreeNode:
    """
    Lightweight record of a dialog's position within a dialog tree.

    Every Dialog owns exactly one DialogTreeNode.  The node carries only ids
    and structural metadata — no message payloads — so the full tree topology
    can be reconstructed from a flat collection (e.g. a log store keyed by
    ``dialog_id``) without loading message content.

    Attributes:
        dialog_id:    Globally unique id for this dialog.
        owner:        Name of the agent that created this dialog.
        parent_id:    dialog_id of the parent (None for root dialogs).
        split_point:  Message index *in the parent* at which this dialog
                      branched.  ``parent.messages[:split_point]`` is the
                      shared prefix that was deep-copied into this child.
        last_n:   The number of the last n messages that are copied from the parent dialog.
        first_k:   The number of the first k messages that are copied from the parent dialog.
        children_ids: dialog_ids of all direct child dialogs.
    """

    dialog_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    owner: Optional[str] = None
    parent_id: Optional[str] = None
    split_point: Optional[int] = None
    children_ids: List[str] = field(default_factory=list)
    last_n: Optional[int] = None
    first_k: Optional[int] = None

    # Live in-process references (not serialized)
    _parent: Optional["DialogTreeNode"] = field(default=None, repr=False)
    _children: List["DialogTreeNode"] = field(default_factory=list, repr=False)

    # -- Lineage helpers --------------------------------------------------

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def depth(self) -> int:
        """Number of forks from the root."""
        d, cur = 0, self
        while cur._parent is not None:
            d += 1
            cur = cur._parent
        return d

    def add_child(self, child: "DialogTreeNode") -> None:
        """Wire a child node into this node (both live refs and id lists)."""
        child._parent = self
        child.parent_id = self.dialog_id
        self._children.append(child)
        if child.dialog_id not in self.children_ids:
            self.children_ids.append(child.dialog_id)

    def subtree_ids(self) -> List[str]:
        """BFS over live children, returning all reachable dialog_ids including self."""
        visited = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            visited.append(node.dialog_id)
            queue.extend(node._children)
        return visited

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, d: Dict[str, Any]) -> "DialogTreeNode":
        return cls(
            dialog_id=d["dialog_id"],
            owner=d.get("owner"),
            parent_id=d.get("parent_id"),
            split_point=d.get("split_point"),
            last_n=d.get("last_n"),
            first_k=d.get("first_k"),
            children_ids=d.get("children_ids", []),
        )


@dataclass
class Dialog:
    """
    An append-only message sequence owned by a single agent.

    The agent that creates a dialog seeds it with its system prompt, and that
    ownership is recorded on the ``tree_node``.  Other participants (user,
    tools, forwarded messages from other agents) append via ``put_*`` helpers,
    but the system-level identity of the dialog never changes.

    **Tree structure** is maintained by a :class:`DialogTreeNode` owned by
    each Dialog.  ``fork()`` creates a child Dialog whose tree_node is
    automatically linked to the parent's tree_node — callers (including
    Agent) never need to wire lineage manually.
    """

    session_name: Optional[str] = None
    top_prompt: Optional[Prompt] = None
    runtime: Optional[Runtime] = None
    owner: Optional[str] = None

    # Message storage (append-only)
    _messages: List[Message] = field(default_factory=list)

    # Tree structure — each Dialog has exactly one node
    tree_node: Optional[DialogTreeNode] = field(default=None)

    # Live Dialog-level parent/children refs (parallel to tree_node's node-level refs)
    _parent_dialog: Optional["Dialog"] = field(default=None, repr=False)
    _children_dialogs: List["Dialog"] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.tree_node is None:
            self.tree_node = DialogTreeNode(owner=self.owner)
        if self.session_name is None:
            self.session_name = (
                dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                + "_"
                + str(uuid.uuid4())[:6]
            )
        self.runtime = self.runtime or get_default_runtime()

    # -- Convenience proxies to tree_node ---------------------------------

    @property
    def dialog_id(self) -> str:
        assert self.tree_node is not None
        return self.tree_node.dialog_id

    @property
    def parent(self) -> Optional["Dialog"]:
        """Live reference to parent Dialog (None for root dialogs)."""
        return self._parent_dialog

    @property
    def children(self) -> List["Dialog"]:
        """Live references to child Dialogs forked from this one."""
        return list(self._children_dialogs)

    @property
    def is_root(self) -> bool:
        assert self.tree_node is not None
        return self.tree_node.is_root

    @property
    def depth(self) -> int:
        assert self.tree_node is not None
        return self.tree_node.depth

    # -- Message access ---------------------------------------------------

    def append(self, message: Message) -> None:
        assert self.tree_node is not None
        message.metadata["dialog_id"] = self.dialog_id
        self._messages.append(message)

    def to_dict(self) -> Dict[str, Any]:
        assert self.tree_node is not None
        return {
            "messages": [message.to_dict() for message in self._messages],
            "session_name": self.session_name,
            "owner": self.owner,
            "tree_node": self.tree_node.to_dict(),
            "top_prompt": self.top_prompt.to_dict() if self.top_prompt else None,
            "top_prompt_path": (
                getattr(self.top_prompt, "_qualified_key", None) or self.top_prompt.path
            )
            if self.top_prompt is not None
            else None,
            "children": [child.to_dict() for child in self._children_dialogs],
        }

    @classmethod
    def from_dict(cls, d: dict, runtime: Optional[Runtime] = None) -> "Dialog":
        from .prompt import Prompt

        top_prompt_data = d.get("top_prompt")
        top_prompt_path = d.get("top_prompt_path")
        runtime = runtime or get_default_runtime()
        top_prompt = Prompt.from_dict(top_prompt_data) if top_prompt_data else None
        if top_prompt is None and top_prompt_path is not None:
            try:
                top_prompt = runtime.get_prompt(top_prompt_path)
            except KeyError:
                logger.warning(
                    "Prompt '%s' not found in runtime during Dialog.from_dict",
                    top_prompt_path,
                )
        tree_node_data = d.get("tree_node")
        tree_node = DialogTreeNode.from_dict(tree_node_data) if tree_node_data else None
        dialog = cls(
            _messages=[Message.from_dict(message) for message in d.get("messages", [])],
            session_name=d.get("session_name"),
            owner=d.get("owner"),
            top_prompt=top_prompt,
            runtime=runtime,
            tree_node=tree_node,
        )
        assert dialog.tree_node is not None
        for child_data in d.get("children", []):
            child = cls.from_dict(child_data, runtime=runtime)
            assert child.tree_node is not None
            child._parent_dialog = dialog
            dialog.tree_node.add_child(child.tree_node)
            dialog._children_dialogs.append(child)
        return dialog

    @property
    def messages(self) -> List[Message]:
        return self._messages

    # -----------------------------------------------------------------------
    # Message Operations, you can only put or fork, dialog is immutable and monotonic.
    # If you want to modify the dialog, you can use ContextManager to dynamically edit it on the fly.
    # -----------------------------------------------------------------------

    # Static/stateless puts

    def put_image(
        self,
        image: Union[str, Path, Any],
        caption: Optional[str] = None,
        name: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        """
        Expects:
        - image: a base64 encoded string, a Path object or string path, or a PIL Image object
        """
        # Resolve string to Path if it looks like a file path
        if isinstance(image, str):
            try:
                p = Path(image)
                if p.exists():
                    image = p
            except (OSError, ValueError):
                pass  # not a path; treat as base64 below
        if isinstance(image, Path):
            with image.open("rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        elif _is_pil_image(image):
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        elif isinstance(image, str):
            # Validate that the base64 string decodes to a known image format
            try:
                raw = base64.b64decode(image)
            except Exception:
                raise ValueError("Invalid base64 encoded image string")
            if _media_type(raw) is None:
                raise ValueError(
                    f"Base64 string does not appear to be a supported image "
                    f"(PNG/JPEG/WebP/GIF); got header {raw[:8]!r}"
                )
            image_base64 = image
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
        payload = _metadata_mapping(metadata)
        if caption is not None:
            payload["caption"] = caption
        message = Message(
            role=role,
            content=image_base64,
            name=name,
            modality=Modalities.IMAGE,
            metadata=payload,
        )
        self.append(message)
        return message

    def put_text(
        self,
        text: str,
        name: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        from .prompt import Prompt  # lazy import to avoid circular dependency

        metadata = _metadata_mapping(metadata)
        # create a temporary prompt for the text to reset parsers and other state
        prompt = Prompt(path="__temp_prompt_" + str(uuid.uuid4())[:6], prompt=text)
        message = Message(
            role=role,
            content=text,
            name=name,
            modality=Modalities.TEXT,
            metadata=metadata,
        )
        self.append(message)
        self.top_prompt = prompt
        return message

    # Stateful put, only prompt can be stateful, other messages are stateless

    def put_prompt(
        self,
        prompt: Prompt | str,
        prompt_args: Optional[Dict[str, Any]] = None,
        name: str = "user",  # or 'user', etc.
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        if isinstance(prompt, str):
            prompt = self.runtime.get_prompt(prompt)
        prompt_args = _optional_mapping("prompt_args", prompt_args)
        metadata = _metadata_mapping(metadata)

        content = prompt(**prompt_args)
        message = Message(
            role=role,
            content=content,
            name=name,
            modality=Modalities.TEXT,
            metadata=metadata,
        )
        self.append(message)
        self.top_prompt = prompt
        return message

    # -----------------------------------------------------------------------
    # Forking — the only way to branch a dialog
    # -----------------------------------------------------------------------

    def fork(self, last_n: int = 0, first_k: int = 1) -> "Dialog":
        """
        Create a child dialog branching from this one.

        Args:
            last_n: if >0, only preserve the last n messages from the parent dialog
                        (useful for retrying from an earlier point).
                        The first system message is always preserved.
            first_k: if >0, ensure the first k messages from the parent dialog
                        (useful for preserving the system prompt message).
                        Should always be >=1 to at least preserve the system prompt message.

        The fork automatically:
        - Deep-copies the message prefix into the child.
        - Creates a child DialogTreeNode linked to this dialog's tree_node.
        - Records split_point on the child's tree_node.
        - Wires live Dialog-level parent/children refs.
        - Inherits session_name, top_prompt, runtime, owner.

        Returns:
            The new child Dialog.
        """
        child = self._copy(last_n, first_k)
        # List the child among this dialog's children (node ids and live refs).
        self.tree_node.add_child(child.tree_node)
        self._children_dialogs.append(child)
        return child

    def _copy(self, last_n: int = 0, first_k: int = 1) -> "Dialog":
        """A fork that names this dialog as its parent but is not its child.

        The agent loop answers each model call on such a working copy: its
        repairs and failed replies are dropped with it when the call ends,
        so ``children``, ``tree_overview()`` and ``to_dict()`` list only
        the forks made with :meth:`fork`.
        """
        last_n = _nonnegative_int_value("last_n", last_n)
        first_k = _nonnegative_int_value("first_k", first_k)
        if last_n >= len(self._messages):
            last_n = 0
        if last_n > 0:
            tail_start = len(self._messages) - last_n
            # Clamp first_k so it doesn't overlap with the tail slice
            first_k = min(first_k, tail_start) if first_k > 0 else 0
            _messages = self._messages[:first_k] + self._messages[tail_start:]
        else:
            _messages = self._messages
        split_point = len(_messages)

        # Build the child's tree node (not yet linked)
        child_tree_node = DialogTreeNode(
            owner=self.owner,
            split_point=split_point,
            last_n=last_n,
            first_k=first_k,
        )

        child = Dialog(
            _messages=[copy.deepcopy(m) for m in _messages],
            session_name=self.session_name,
            top_prompt=self.top_prompt,
            runtime=self.runtime,
            owner=self.owner,
            tree_node=child_tree_node,
        )
        child_tree_node.parent_id = self.dialog_id
        child_tree_node._parent = self.tree_node
        child._parent_dialog = self
        return child

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------

    def overview(
        self,
        remove_tail: bool = False,
        max_length: int = 100,
        stream=None,
        divider: bool = False,
    ):
        remove_tail = _bool_value("remove_tail", remove_tail)
        max_length = _nonnegative_int_value("max_length", max_length)
        _overview = ""
        for idx, message in enumerate(self.messages):
            if remove_tail and idx == len(self.messages) - 1:
                break
            text = (
                _image_placeholder(message)
                if message.modality == Modalities.IMAGE
                else str(message.content)
            )
            content_preview = (
                text[:max_length] + "..." if len(text) > max_length else text
            )
            _overview += f"[{idx}. {message.name} ({message.role.msg_value})]: {content_preview}\n\n"

        _overview = _overview.strip()
        cost = self.tail.cost if self.messages else InvokeCost()
        if stream is not None:
            if divider:
                stream.divider()
            stream.write(
                U.html_collapse("Context overview", _overview), unsafe_allow_html=True
            )
            stream.write(str(cost))
        return _overview

    def tree_overview(self, indent: int = 0) -> str:
        """
        Recursively print the dialog tree structure from this node.
        Useful for debugging multi-fork scenarios.
        """
        indent = _nonnegative_int_value("indent", indent)
        prefix = "  " * indent
        branch = "└─ " if indent > 0 else ""
        line = (
            f"{prefix}{branch}[{self.dialog_id[:8]}] "
            f"owner={self.owner} msgs={len(self._messages)} "
            f"split@{self.tree_node.split_point}"
        )
        if self.tree_node.last_n is not None and self.tree_node.last_n > 0:
            line += (
                f" (last_n={self.tree_node.last_n}, first_k={self.tree_node.first_k})"
            )
        lines = [line]
        for child in self._children_dialogs:
            lines.append(child.tree_overview(indent + 1))
        return "\n".join(lines)

    @property
    def tail(
        self,
    ):  # last message in the dialog, use it to get last response from the LLM
        return self._messages[-1] if self._messages else None

    @property
    def head(self):  # usually the system prompt message
        return self._messages[0] if self._messages else None

    @property
    def cost(self) -> InvokeCost:
        costs = [message.cost for message in self._messages]
        return InvokeCost(
            prompt_tokens=sum(c.prompt_tokens for c in costs),
            completion_tokens=sum(c.completion_tokens for c in costs),
            total_tokens=sum(c.total_tokens for c in costs),
            cached_prompt_tokens=sum(c.cached_prompt_tokens for c in costs),
            reasoning_tokens=sum(c.reasoning_tokens for c in costs),
            audio_prompt_tokens=sum(c.audio_prompt_tokens for c in costs),
            audio_completion_tokens=sum(c.audio_completion_tokens for c in costs),
            # We don't aggregate token rates for the whole dialog
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            cache_read_input_token_cost=0.0,
            # Aggregate absolute dollar values
            prompt_cost=sum(c.prompt_cost for c in costs),
            completion_cost=sum(c.completion_cost for c in costs),
            cost=sum(c.cost for c in costs),
        )


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


#: Context windows found, by model; unknown ones are looked up again.
_CONTEXT_WINDOWS: Dict[str, int] = {}
#: Models whose unknown window was warned about.
_UNKNOWN_WINDOWS: set = set()


def _model_info(model_name: str) -> Dict[str, Any]:
    """LiteLLM's model info, or {} for a model it does not know.

    Looked up by the name without its provider prefix: for a Hugging Face
    model, LiteLLM builds the hub URL from that name, and the prefix would
    make it a 404. A name whose prefix is no provider (a model registered
    as ``pkg/model``) is looked up whole.
    """
    import litellm

    try:
        name, provider, _, _ = litellm.get_llm_provider(model_name)
        return litellm.get_model_info(name, custom_llm_provider=provider)
    except Exception:
        pass
    try:
        return litellm.get_model_info(model_name)
    except Exception:
        return {}


def _context_window(model_name: str) -> Optional[int]:
    """The model's context window from LiteLLM's model info, or None.

    That is ``max_input_tokens``; a model registered with
    ``litellm.register_model`` should give it. For a Hugging Face model,
    LiteLLM reports the hub's ``max_position_embeddings`` as ``max_tokens``
    alone, and that is the window; elsewhere ``max_tokens`` can be the
    output limit, so it is not used. A window found is remembered. An
    unknown one is warned about once and looked up again next time, so a
    model registered later is found.
    """
    if model_name in _CONTEXT_WINDOWS:
        return _CONTEXT_WINDOWS[model_name]
    info = _model_info(model_name)
    window = info.get("max_input_tokens")
    if (
        not window
        and info.get("litellm_provider") == "huggingface"
        and not info.get("max_output_tokens")
    ):
        window = info.get("max_tokens")
    if window:
        _CONTEXT_WINDOWS[model_name] = int(window)
        return int(window)
    if model_name not in _UNKNOWN_WINDOWS:
        _UNKNOWN_WINDOWS.add(model_name)
        logger.warning(
            "DefaultContextManager: LiteLLM does not know the context window "
            "of %r, so its calls are not trimmed. Set max_tokens to trim them.",
            model_name,
        )
    return None


def _trim_units(messages: List[Message]) -> List[List[int]]:
    """Indexes of ``messages[1:]``, newest first, in the groups trimming keeps
    or drops whole: a tool call with its results, or a single message.

    A tool result sent without the call it answers is rejected by
    OpenAI-compatible APIs.
    """
    units: List[List[int]] = []
    results: List[int] = []
    for index in range(len(messages) - 1, 0, -1):
        message = messages[index]
        if message.role == Roles.TOOL:
            results.insert(0, index)
            continue
        if message.role == Roles.TOOL_CALL or message.is_function_call:
            units.append([index, *results])
        else:
            if results:  # results whose call is not in the dialog
                units.append(results)
            units.append([index])
        results = []
    if results:
        units.append(results)
    return units


class ContextManager(ABC):
    """
    Pre-processing hook applied to a Dialog before it is sent to the LLM.

    The Agent calls ``context_manager(working_dialog)`` once per LLM turn,
    right after forking the canonical dialog, and uses the returned dialog for
    the actual invoke.  The original canonical dialog is never mutated.

    **Contract**

    ``__call__(dialog)`` receives the (already-forked) working dialog and must
    return a Dialog that is safe to pass to the invoker.  It may return the
    same object unchanged, or a new child dialog produced via ``dialog.fork()``
    or a fresh ``Dialog`` instance.

    **Built-in implementation**

    :class:`DefaultContextManager` — drops / truncates old messages so the
    total token count stays within the model's context window.

    **Extending**

    Subclass ``ContextManager``, set the ``name`` class attribute, and
    implement ``__call__``::

        class SummaryCompressor(ContextManager):
            name = "summary"   # <-- required; used as the config type key

            def __init__(self, model_name: str, max_tokens: Optional[int] = None):
                self.model_name = model_name
                self._max_tokens = max_tokens

            def __call__(self, dialog: Dialog) -> Dialog:
                # ... compress middle messages into a rolling summary ...
                return compressed_dialog

    Register the class with the runtime so config can reference it by name.
    ``register_context_manager`` reads ``cls.name`` automatically::

        runtime.register_context_manager(SummaryCompressor)
        # equivalent to registering under the key "summary"

    Then in your tactic YAML config::

        global:
          context_manager:
            type: summary   # resolved via runtime registry
            max_tokens: 128000

        agent_configs:
          - name: my_agent
            model_name: gpt-4o
            context_manager:          # per-agent override (deep-merged with global)
              type: default
              max_tokens: 32000

    Set ``type: null`` (or omit the key entirely) to disable context management
    for a specific agent even when a global default is configured.

    **Programmatic usage** (without config)::

        from lllm.runtimes.native import DefaultContextManager

        cm = DefaultContextManager("gpt-4o")
        agent = Agent(..., context_manager=cm)
    """

    #: Registry key used by :meth:`~lllm.core.runtime.Runtime.register_context_manager`
    #: and the ``type`` field in YAML config.  Must be set on every concrete subclass.
    name: ClassVar[str]

    @abstractmethod
    def __call__(self, dialog: Dialog) -> Dialog:
        pass


class DefaultContextManager(ContextManager):
    """
    Token-aware dialog truncator — the built-in :class:`ContextManager`.

    Uses ``litellm.token_counter`` for accurate, model-specific token counting
    and drops or truncates the oldest non-system messages when the dialog
    exceeds the token budget: ``max_tokens``, or else the model's context
    window from LiteLLM's model info (``max_input_tokens``, or the hub's
    context length for a Hugging Face model). For a model LiteLLM does not
    know, such as one on your own server, set ``max_tokens`` (or register
    the model with ``litellm.register_model``, giving ``max_input_tokens``);
    without it nothing is trimmed (and a warning is logged).

    **Truncation strategy**

    Messages are counted as they are sent: an image as its caption and an
    ``image_url`` part (about a hundred tokens for a photo), not as its data.

    1. The first message (system / developer prompt) is always preserved.
    2. The newest message is always sent whole, with a warning if it alone
       exceeds the budget.
    3. Older messages are retained from the **tail** (most recent) inward
       until the token budget is exhausted. A tool call and its results are
       kept or dropped together: APIs reject a result without its call.
    4. The *border* message — the oldest message that partially fits — keeps
       its end, the part closest to the kept messages, after a
       ``[...earlier content truncated...]`` marker. Images and other
       non-text messages are never cut: an image that does not fit is
       dropped, with a warning.
    5. A ``SAFETY_BUFFER`` (default 5 000 tokens, or a quarter of a budget
       under 20 000) is subtracted from the budget, leaving room for the
       reply, so the invoker never lands right at the edge.

    **Quickstart**::

        from lllm.runtimes.native import DefaultContextManager

        cm = DefaultContextManager("gpt-4o")          # the model's window
        cm = DefaultContextManager("gpt-4o", max_tokens=32000)  # manual cap

        agent = Agent(..., context_manager=cm)

    **Config (YAML)** — ``type: default`` resolves to this class without any
    runtime registration (it is the built-in)::

        # global default — applies to every agent in the tactic
        global:
          context_manager:
            type: default       # built-in; no registration needed
            max_tokens: 128000  # optional; omit to auto-detect from litellm

        agent_configs:
          - name: cheap_agent
            model_name: gpt-4o-mini
            context_manager:    # per-agent override (deep-merged with global)
              type: default
              max_tokens: 16000

          - name: no_cm_agent
            model_name: gpt-4o
            context_manager:    # explicitly disable for this agent
              type: null
    """

    name: ClassVar[str] = "default"
    SAFETY_BUFFER: int = 5000

    def __init__(self, model_name: str, max_tokens: Optional[int] = None):
        """
        Args:
            model_name: litellm-compatible model name (e.g. "gpt-4o").
            max_tokens:  Token budget for what is sent, including the
                         ``SAFETY_BUFFER``.  When omitted, the model's
                         context window is looked up in LiteLLM.
        """
        self.model_name = model_name
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def max_tokens(self) -> Optional[int]:
        """``max_tokens`` if given, else the model's context window.

        None when LiteLLM does not know the model's window; then nothing
        is trimmed.
        """
        if self._max_tokens is not None:
            return self._max_tokens
        return _context_window(self.model_name)

    def _to_raw(self, m: Message) -> dict:
        """A message as litellm's token_counter should see it: as it is sent.

        An image is its caption and an ``image_url`` part, as the invoker
        sends it, so it counts as an image (about a hundred tokens for a
        photo), not as its base64 text.
        """
        if m.modality == Modalities.IMAGE and isinstance(m.content, str):
            content: Any = _image_parts(m)
        elif isinstance(m.content, str):
            content = m.content
        else:
            content = str(m.content)
        return {"role": m.role.value, "content": content}

    def _count(self, raw_msgs: list) -> int:
        from litellm import token_counter

        return token_counter(model=self.model_name, messages=raw_msgs)

    @staticmethod
    def _cut(message: Message, budget: int, tokens: int) -> Optional[Message]:
        """The end of a text message that only partly fits, after a marker.

        None when the budget left is too small, or for an image, whose data
        cannot be cut.
        """
        if budget <= 50 or message.modality != Modalities.TEXT:
            return None
        if not isinstance(message.content, str):
            return None
        content = message.content
        # Approximate character count via token ratio (0.85 headroom)
        keep = max(1, int(len(content) * budget / tokens * 0.85))
        border = copy.deepcopy(message)
        border.content = "[...earlier content truncated...]\n" + content[-keep:]
        return border

    # ------------------------------------------------------------------
    # Core truncation
    # ------------------------------------------------------------------

    def __call__(self, dialog: Dialog) -> Dialog:
        messages = dialog.messages
        if not messages:
            return dialog

        max_tokens = self.max_tokens
        if max_tokens is None:  # unknown window: nothing to trim against
            return dialog
        # Room for the reply: SAFETY_BUFFER, or a quarter of a small budget.
        limit = max_tokens - min(self.SAFETY_BUFFER, max_tokens // 4)
        raw_all = [self._to_raw(m) for m in messages]

        if self._count(raw_all) <= limit:
            return dialog

        logger.debug(
            "DefaultContextManager: dialog %s exceeds token limit, truncating",
            dialog.dialog_id,
        )

        # Always keep the first (system) message
        head_tokens = self._count([raw_all[0]])
        budget = limit - head_tokens

        # Walk from the tail inward; collect messages that fit within budget.
        # A tool call goes with its results, and the newest is sent whole.
        kept: List[Message] = []
        kept_indexes = set()
        for position, unit in enumerate(_trim_units(messages)):
            t = sum(self._count([raw_all[index]]) for index in unit)
            if t <= budget or position == 0:
                if t > budget:
                    logger.warning(
                        "DefaultContextManager: the newest message of dialog %s "
                        "needs %d tokens, more than the %d left; it is sent whole.",
                        dialog.dialog_id,
                        t,
                        budget,
                    )
                kept[:0] = [messages[index] for index in unit]
                kept_indexes.update(unit)
                budget -= t
                continue
            if len(unit) == 1:
                border = self._cut(messages[unit[0]], budget, t)
                if border is not None:
                    kept.insert(0, border)
                    kept_indexes.add(unit[0])
            break

        dropped_images = sum(
            1
            for index in range(1, len(messages))
            if index not in kept_indexes
            and messages[index].modality == Modalities.IMAGE
        )
        if dropped_images:
            logger.warning(
                "DefaultContextManager dropped %d image(s) from dialog %s to "
                "fit its %d-token budget.",
                dropped_images,
                dialog.dialog_id,
                limit,
            )

        new_messages = [copy.deepcopy(messages[0])] + kept

        # Build child dialog wired into the dialog tree
        child_node = DialogTreeNode(
            owner=dialog.owner,
            split_point=len(new_messages),
        )
        child = Dialog(
            _messages=new_messages,
            session_name=dialog.session_name,
            top_prompt=dialog.top_prompt,
            runtime=dialog.runtime,
            owner=dialog.owner,
            tree_node=child_node,
        )
        dialog.tree_node.add_child(child.tree_node)
        child._parent_dialog = dialog
        dialog._children_dialogs.append(child)
        return child
