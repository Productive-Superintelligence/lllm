from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from lllm.core.const import APITypes, InvokeResult
from lllm.core.dialog import Dialog



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


class BaseInvoker(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

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