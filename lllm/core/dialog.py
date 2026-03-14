import uuid
import base64
from PIL import Image
import datetime as dt
from pathlib import Path
import copy
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator

from lllm.core.prompt import Prompt, InvokeCost, FunctionCall
from lllm.core.const import Roles, Modalities, RCollections, APITypes
from lllm.core.log import ReplayableLogBase
import lllm.utils as U
from lllm.core.runtime import Runtime, get_default_runtime
import logging
logger = logging.getLogger(__name__)



class TokenLogprob(BaseModel):
    token: Optional[str] = None
    logprob: Optional[float] = None
    bytes: Optional[List[int]] = None
    top_logprobs: List['TokenLogprob'] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

TokenLogprob.model_rebuild()

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def _sanitize_name(raw_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)[:64]

class Message(BaseModel):
    role: Roles
    content: Union[str, List[Dict[str, Any]]] # Content can be string or list of content parts (for images)
    name: str # name of the sender
    function_calls: List[FunctionCall] = Field(default_factory=list)
    modality: Modalities = Modalities.TEXT
    logprobs: List[TokenLogprob] = Field(default_factory=list)
    parsed: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict) # 
    metadata: Dict[str, Any] = Field(default_factory=dict) 
    api_type: APITypes = APITypes.COMPLETION

    vectors: List[float] = Field(default_factory=list) # place holder for embedding vectors of the message, can be used for training, similarity search, etc. Need special invoker to support this.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def sanitized_name(self):
        return _sanitize_name(self.name)

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

        p_tokens = self.usage.get('prompt_tokens', 0)
        c_tokens = self.usage.get('completion_tokens', 0)
        t_tokens = self.usage.get('total_tokens', p_tokens + c_tokens)
        p_details = self.usage.get('prompt_tokens_details', {}) or {}
        c_details = self.usage.get('completion_tokens_details', {}) or {}
    
        return InvokeCost(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            cached_prompt_tokens=p_details.get('cached_tokens', 0),
            audio_prompt_tokens=p_details.get('audio_tokens', 0),
            reasoning_tokens=c_details.get('reasoning_tokens', 0),
            audio_completion_tokens=c_details.get('audio_tokens', 0),
            # Dollar costs
            input_cost_per_token=self.usage.get("input_cost_per_token", 0.0),
            output_cost_per_token=self.usage.get("output_cost_per_token", 0.0),
            cache_read_input_token_cost=self.usage.get("cache_read_input_token_cost", 0.0),
            prompt_cost=self.usage.get("prompt_cost", 0.0),
            completion_cost=self.usage.get("completion_cost", 0.0),
            cost=self.usage.get("response_cost", 0.0)
        )

    @property
    def is_function_call(self) -> bool:
        return len(self.function_calls) > 0

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

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
    _parent: Optional['DialogTreeNode'] = field(default=None, repr=False)
    _children: List['DialogTreeNode'] = field(default_factory=list, repr=False)

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

    def add_child(self, child: 'DialogTreeNode') -> None:
        """Wire a child node into this node (both live refs and id lists)."""
        child._parent = self
        child.parent_id = self.dialog_id
        self._children.append(child)
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
            'dialog_id': self.dialog_id,
            'owner': self.owner,
            'parent_id': self.parent_id,
            'split_point': self.split_point,
            'children_ids': list(self.children_ids),
            'last_n': self.last_n,
            'first_k': self.first_k,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DialogTreeNode':
        return cls(
            dialog_id=d['dialog_id'],
            owner=d.get('owner'),
            parent_id=d.get('parent_id'),
            split_point=d.get('split_point'),
            last_n=d.get('last_n'),
            first_k=d.get('first_k'),
            children_ids=d.get('children_ids', []),
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
    session_name: str = None
    log_base: Optional[ReplayableLogBase] = None
    top_prompt: Optional[Prompt] = None
    runtime: Optional[Runtime] = None
    owner: Optional[str] = None

    # Message storage (append-only)
    _messages: List[Message] = field(default_factory=list)

    # Tree structure — each Dialog has exactly one node
    tree_node: DialogTreeNode = field(default=None)

    # Live Dialog-level parent/children refs (parallel to tree_node's node-level refs)
    _parent_dialog: Optional['Dialog'] = field(default=None, repr=False)
    _children_dialogs: List['Dialog'] = field(default_factory=list, repr=False)


    def __post_init__(self):
        if self.tree_node is None:
            self.tree_node = DialogTreeNode(owner=self.owner)
        if self.session_name is None:
            self.session_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:6]
        self.runtime = self.runtime or get_default_runtime()
        if self.log_base is not None:
            dialogs_sess = self.log_base.get_collection(RCollections.DIALOGS).create_session(self.session_name)
            dialogs_sess.log(self.dialog_id, metadata=self.tree_node.to_dict())
            self.sess = self.log_base.get_collection(RCollections.MESSAGES).create_session(
                f'{self.session_name}/{self.dialog_id}'
            )
        else:
            self.sess = None

    # -- Convenience proxies to tree_node ---------------------------------

    @property
    def dialog_id(self) -> str:
        return self.tree_node.dialog_id

    @property
    def parent(self) -> Optional['Dialog']:
        """Live reference to parent Dialog (None for root dialogs)."""
        return self._parent_dialog

    @property
    def children(self) -> List['Dialog']:
        """Live references to child Dialogs forked from this one."""
        return list(self._children_dialogs)

    @property
    def is_root(self) -> bool:
        return self.tree_node.is_root

    @property
    def depth(self) -> int:
        return self.tree_node.depth

    # -- Message access ---------------------------------------------------

    def append(self, message: Message): # ensure this is the only way to write the messages to make sure the trackability
        message.metadata['dialog_id'] = self.dialog_id
        self._messages.append(message)
        if self.sess is not None:
            try:
                self.sess.log(message.content, metadata=message.to_dict()) # Use to_dict for logging
            except Exception as e:
                print(f'WARNING: Failed to log message: {e}, log the message without metadata')
                self.sess.log(message.content)

    def to_dict(self):
        return {
            'messages': [message.to_dict() for message in self._messages],
            'session_name': self.session_name,
            'owner': self.owner,
            'tree_node': self.tree_node.to_dict(),
            'top_prompt_path': self.top_prompt.path if self.top_prompt is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, log_base: ReplayableLogBase = None, runtime: Runtime = None):
        top_prompt_path = d.get('top_prompt_path')
        runtime = runtime or get_default_runtime()
        top_prompt = None
        if top_prompt_path is not None:
            try:
                top_prompt = runtime.get_prompt(top_prompt_path)
            except KeyError:
                logger.warning("Prompt '%s' not found in runtime during Dialog.from_dict", top_prompt_path)
        tree_node_data = d.get('tree_node')
        tree_node = DialogTreeNode.from_dict(tree_node_data) if tree_node_data else None
        return cls(
            _messages=[Message.from_dict(message) for message in d['messages']],
            log_base=log_base,
            session_name=d['session_name'],
            owner=d.get('owner'),
            top_prompt=top_prompt,
            runtime=runtime,
            tree_node=tree_node,
        )
    
    @property
    def messages(self):
        return self._messages
    
    # -----------------------------------------------------------------------
    # Message Operations, you can only put or fork, dialog is immutable and monotonic.
    # If you want to modify the dialog, you can use ContextManager to dynamically edit it on the fly.
    # -----------------------------------------------------------------------

    # Static/stateless puts

    def put_image(
        self,
        image: Union[str, Path, Image.Image],
        caption: str = None,
        name: str = 'user',
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        """
        Expects:
        - image: a base64 encoded string, a Path object or string path, or a PIL Image object
        """
        if Path(image).exists():
            # interpret the string as a file path
            image = Path(image)
        if isinstance(image, Path):
            with image.open('rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, Image.Image):
            image_base64 = base64.b64encode(image.tobytes()).decode('utf-8')
        elif isinstance(image, str):
            # check if the string is a valid base64 encoded string
            if not base64.b64decode(image).startswith(b'\x89PNG'):
                raise ValueError(f'Invalid base64 encoded string: {image}')
            image_base64 = image
        else:
            raise ValueError(f'Invalid image type: {type(image)}')
        payload = dict(metadata) if metadata else {}
        if caption is not None:
            payload['caption'] = caption
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
        name: str = 'user',
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        metadata = dict(metadata) if metadata else {}
        # create a temporary prompt for the text to reset parsers and other state
        prompt = Prompt(path='__temp_prompt_'+str(uuid.uuid4())[:6], prompt=text)
        message = Message(
            role=role,
            content=text,
            name=name,
            modality=Modalities.TEXT,
            metadata=metadata
        )
        self.append(message)
        self.top_prompt = prompt
        return message

    # Stateful put, only prompt can be stateful, other messages are stateless

    def put_prompt(
        self,
        prompt: Prompt | str,
        prompt_args: Optional[Dict[str, Any]] = None,
        name: str = 'user',  # or 'user', etc.
        metadata: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        if isinstance(prompt, str):
            prompt = self.runtime.get_prompt(prompt)
        prompt_args = dict(prompt_args) if prompt_args else {}
        metadata = dict(metadata) if metadata else {}
        if not prompt_args:
            content = prompt.prompt
        else:
            content = prompt(**prompt_args)
        message = Message(
            role=role,
            content=content,
            name=name,
            modality=Modalities.TEXT,
            metadata=metadata
        )
        self.append(message)
        self.top_prompt = prompt
        return message
    
    # -----------------------------------------------------------------------
    # Forking — the only way to branch a dialog
    # -----------------------------------------------------------------------
    
    def fork(self, last_n: int = 0, first_k: int = 1) -> 'Dialog':
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
        - Inherits session_name, log_base, top_prompt, runtime, owner.

        Returns:
            The new child Dialog.
        """
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
            log_base=self.log_base,
            top_prompt=self.top_prompt,
            runtime=self.runtime,
            owner=self.owner,
            tree_node=child_tree_node,
        )

        # Wire tree_node parent ↔ child (sets parent_id, live refs, children_ids)
        self.tree_node.add_child(child.tree_node)

        # Wire Dialog-level live refs
        child._parent_dialog = self
        self._children_dialogs.append(child)

        return child

    # -----------------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------------

    def overview(self, remove_tail: bool = False, max_length: int = 100, 
                 stream = None, divider: bool = False):
        _overview = ''
        for idx, message in enumerate(self.messages):
            if remove_tail and idx == len(self.messages)-1:
                break
            content_preview = str(message.content)[:max_length] + '...' if len(str(message.content)) > max_length else str(message.content)
            _overview += f'[{idx}. {message.name} ({message.role.msg_value})]: {content_preview}\n\n'
        
        _overview = _overview.strip()
        cost = self.tail.cost if self.messages else InvokeCost()
        if stream is not None:
            if divider:
                stream.divider()
            stream.write(U.html_collapse(f'Context overview', _overview), unsafe_allow_html=True)
            stream.write(str(cost))
        return _overview

    def tree_overview(self, indent: int = 0) -> str:
        """
        Recursively print the dialog tree structure from this node.
        Useful for debugging multi-fork scenarios.
        """
        prefix = '  ' * indent
        branch = '└─ ' if indent > 0 else ''
        line = (
            f"{prefix}{branch}[{self.dialog_id[:8]}] "
            f"owner={self.owner} msgs={len(self._messages)} "
            f"split@{self.tree_node.split_point}"
        )
        if self.tree_node.last_n is not None and self.tree_node.last_n > 0:
            line += f" (last_n={self.tree_node.last_n}, first_k={self.tree_node.first_k})"
        lines = [line]
        for child in self._children_dialogs:
            lines.append(child.tree_overview(indent + 1))
        return '\n'.join(lines)

    @property
    def tail(self): # last message in the dialog, use it to get last response from the LLM
        return self._messages[-1] if self._messages else None
    
    @property
    def head(self): # usually the system prompt message
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
            cost=sum(c.cost for c in costs)
        )



# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


class ContextManager(ABC):
    """
    Context manager for the dialog, it can be truncator, compressor, memory manager, editor, etc.

    __call__:
        Expects: a raw dialog as input
        Returns: the new dialog that fit the agent context limit or whatever policy is desired
    The policy should be setup by __init__ method through the user.

    For example:
    - A truncator may just need to be aware of context limit and truncate the dialog accordingly.
    - A compressor is similar but work in a more sophisticated way that use like agentic compression techniques when hit the context limit.
    - A memory manager may need to store the memory for each dialog in a dict then update it on the fly by detecting the changes.
    - An editor may edit the dialog in a more sophisticated way that use like agentic editing techniques.
    """
    @abstractmethod
    def __call__(self, dialog: Dialog) -> Dialog:
        pass


class DefaultLiteLLMTruncator(ContextManager):
    """
    Default context manager that truncate the dialog to the last n messages.

    It assumes the LiteLLM invoker is used where we can get the context limit and the tokenizer. 
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        raise NotImplementedError("This method is not implemented yet")

    def __call__(self, dialog: Dialog) -> Dialog:
        raise NotImplementedError("This method is not implemented yet")

