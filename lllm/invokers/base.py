from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lllm.core.const import APITypes
from lllm.core.dialog import Dialog, Message


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
    message: Optional[Message] = None  # always set by invoker, None is just the dataclass default

    @property
    def has_errors(self) -> bool:
        return len(self.execution_errors) > 0

    @property
    def error_message(self) -> str:
        return '\n'.join(str(e) for e in self.execution_errors)


class BaseInvoker(ABC):
    @abstractmethod
    def call(
        self,
        dialog: Dialog,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        metadata: Optional[Dict[str, Any]] = None, # only for tracking additional information, such as frontend replay info
        api_type: APITypes = APITypes.COMPLETION,
        stream_handler: BaseStreamHandler = None,
    ) -> InvokeResult:
        """
        Call the LLM and return the invocation result.
        """
        pass


class BaseStreamHandler(ABC):
    @abstractmethod
    def handle_chunk(self, chunk_content: str, chunk_response: Any):
        """
        Handle a chunk of the streaming response.
        Args:
            chunk_content, str: The content of the chunk.
            chunk_response, Any: The raw response object. Please refer to https://docs.litellm.ai/docs/#streaming for more details. 
             - Note that the object for response API and completion API are different.

        Example usage:
        ```python
        print(chunk_content) # or like link to your frontend display
        ```
        You can also log other things you are interested in in the object, but you need to handle the state yourself.
        """
        pass