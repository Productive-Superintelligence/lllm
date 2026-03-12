import uuid
import copy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from lllm.core.models import Message, Prompt, InvokeCost
from lllm.core.const import Roles, Modalities, RCollections
from lllm.core.log import ReplayableLogBase
import lllm.utils as U
from lllm.core.context import Context, get_default_context

@dataclass
class Dialog:
    """
    Whenever a dialog is created/forked, it should be associated with a session name
    """
    _messages: List[Message]
    session_name: str
    log_base: Optional[ReplayableLogBase] = None
    parent_dialog: Optional[str] = None
    top_prompt: Optional[Prompt] = None

    def __post_init__(self):
        self.dialog_id = uuid.uuid4().hex
        if self.log_base is not None:
            dialogs_sess = self.log_base.get_collection(RCollections.DIALOGS).create_session(self.session_name) # track the dialogs created in this session
            dialogs_sess.log(self.dialog_id, metadata={'parent_dialog': self.parent_dialog})
            self.sess = self.log_base.get_collection(RCollections.MESSAGES).create_session(f'{self.session_name}/{self.dialog_id}') # track the dialogs created in this session
        else:
            self.sess = None

    def append(self, message: Message): # ensure this is the only way to write the messages to make sure the trackability
        message.extra['dialog_id'] = self.dialog_id
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
            'parent_dialog': self.parent_dialog,
            'top_prompt_path': self.top_prompt.path if self.top_prompt is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, log_base: ReplayableLogBase, context: Context = None):
        top_prompt_path = d['top_prompt_path']
        context = context or get_default_context()
        if top_prompt_path is not None:
            top_prompt = context.get_prompt(top_prompt_path)
            if top_prompt is None:
                 print(f"Warning: Prompt {top_prompt_path} not found in context.")
        else:
            top_prompt = None
        return cls(
            _messages=[Message.from_dict(message) for message in d['messages']],
            log_base=log_base,
            session_name=d['session_name'],
            parent_dialog=d['parent_dialog'],
            top_prompt=top_prompt,
        )
    
    @property
    def messages(self):
        return self._messages
    
    def send_base64_image(
        self,
        image_base64: str,
        caption: str = None,
        name: str = 'user',
        extra: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        payload = dict(extra) if extra else {}
        if caption is not None:
            payload['caption'] = caption
        message = Message(
            role=role,
            content=image_base64,
            name=name,
            modality=Modalities.IMAGE,
            extra=payload,
        )
        self.append(message)
        return message

    def send_message(
        self,
        prompt: Prompt | str,
        prompt_args: Optional[Dict[str, Any]] = None,
        name: str = 'user',  # or 'user', etc.
        extra: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ) -> Message:
        prompt_args = dict(prompt_args) if prompt_args else {}
        metadata = dict(extra) if extra else {}
        if isinstance(prompt, str):
            assert not prompt_args, "Prompt args are not allowed for string prompt"
            # Create a temporary prompt object
            prompt = Prompt(path='__temp_prompt_'+str(uuid.uuid4())[:6], prompt=prompt)
            content = prompt.prompt
        elif not prompt_args:
            content = prompt.prompt
        else:
            content = prompt(**prompt_args)
        message = Message(
            role=role,
            content=content,
            name=name,
            modality=Modalities.TEXT,
            extra=metadata
        )
        self.append(message)
        self.top_prompt = prompt
        return message
    
    def fork(self) -> 'Dialog':
        _messages = [copy.deepcopy(message) for message in self._messages]
        _dialog = Dialog(
            _messages=_messages,
            session_name=self.session_name,
            log_base=self.log_base,
            parent_dialog=self.dialog_id,
            top_prompt=self.top_prompt,
        )
        return _dialog
    
    def overview(self, remove_tail: bool = False, max_length: int = 100, 
                 stream = None, divider: bool = False):
        _overview = ''
        for idx, message in enumerate(self.messages):
            if remove_tail and idx == len(self.messages)-1:
                break
            # message.overview() logic needs to be in Message class or here
            # Message class has overview method in original code, I should add it back to Message model if I missed it
            # I missed it in Message model. I'll implement a simple one here or add it to Message.
            # Let's assume Message has it or I implement it here.
            # Implementing here for safety if I missed it in Pydantic model.
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

    @property
    def tail(self): # last message in the dialog, use it to get last response from the LLM
        return self._messages[-1] if self._messages else None
    
    @property
    def system(self):
        return self._messages[0] if self._messages else None

    def context_copy(self, n: int = 1) -> 'Dialog':
        _dialog = self.fork()
        if n > 0:
            _dialog._messages = _dialog._messages[:-n]
        return _dialog

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