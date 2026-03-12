import os
import json
import re
import logging
from typing import Any, Dict, List, Optional

from lllm.core.models import Message, Prompt, FunctionCall, TokenLogprob
from lllm.core.const import Roles, Modalities, APITypes, Invokers
from lllm.invokers.base import BaseInvoker, BaseStreamHandler
from lllm.core.dialog import Dialog

from litellm import stream_chunk_builder
from litellm import completion as completion_api
from litellm import responses as responses_api

logger = logging.getLogger(__name__)

PROVIDER_APIKEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "HUGGINGFACE_API_KEY",
    "OPENROUTER_API_KEY",
    "NOVITA_API_KEY",
    "VERCEL_AI_GATEWAY_API_KEY",
]

NVIDIA_NIM_REQUIRED_ENV_VARS = [
    "NVIDIA_NIM_API_KEY",
    "NVIDIA_NIM_API_BASE",
]

VERTEXAI_REQUIRED_ENV_VARS = [
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
]

AZURE_REQUIRED_ENV_VARS = [
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
]

ALL_ENV_VARS = [
    *VERTEXAI_REQUIRED_ENV_VARS,
    *NVIDIA_NIM_REQUIRED_ENV_VARS,
    *AZURE_REQUIRED_ENV_VARS,
    *PROVIDER_APIKEYS,
]


def _check_required_env_vars(required_env_vars: List[str], tag):
    if any(env_var in os.environ for env_var in required_env_vars):
        if not all(env_var in os.environ for env_var in required_env_vars):
            raise ValueError(f"Missing required environment variables for {tag}: {required_env_vars}")
        return True
    return False

def _check_env_vars():
    _check_required_env_vars(VERTEXAI_REQUIRED_ENV_VARS, "VERTEXAI")
    _check_required_env_vars(NVIDIA_NIM_REQUIRED_ENV_VARS, "NVIDIA_NIM")
    _check_required_env_vars(AZURE_REQUIRED_ENV_VARS, "AZURE")

    if all(env_var not in os.environ for env_var in ALL_ENV_VARS):
        logger.warning("No environment variables found for any provider. Ignore if you are using Ollama provider.")

_check_env_vars()




class LiteLLMInvoker(BaseInvoker):
    
    def _convert_dialog(self, dialog: Dialog) -> List[Dict[str, str]]:
        """Convert internal Dialog state into OpenAI-compatible messages."""
        messages: List[Dict[str, str]] = []
        for message in dialog.messages:
            if message.role in (Roles.ASSISTANT, Roles.TOOL_CALL):
                assistant_entry: Dict[str, str] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.name and message.name not in ("assistant", "user", "system", "internal"):
                    assistant_entry["name"] = message.sanitized_name
                if message.function_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": fc.id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.arguments),
                            },
                        }
                        for fc in message.function_calls
                    ]
                messages.append(assistant_entry)
                continue

            if message.role == Roles.TOOL:
                tool_call_id = message.metadata.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError(
                        "Tool call id is not found in the message metadata for tool message: "
                        f"{message}"
                    )
                tool_entry = {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": tool_call_id,
                }
                if message.name and message.name not in ("assistant", "user", "system", "internal"):
                    tool_entry["name"] = message.sanitized_name
                messages.append(tool_entry)
                continue

            if message.modality == Modalities.IMAGE:
                content_parts = []
                if "caption" in message.metadata:
                    content_parts.append({"type": "text", "text": message.metadata["caption"]})
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{message.content}"}}
                )
                messages.append({"role": message.role.msg_value, "content": content_parts, "name": message.sanitized_name})
                continue

            if message.modality == Modalities.TEXT:
                messages.append({"role": message.role.msg_value, "content": message.content, "name": message.sanitized_name})
                continue

            raise ValueError(f"Unsupported modality: {message.modality}")

        return messages

    def _build_tools(self, prompt: Prompt) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for func in prompt.functions.values():
            tool = func.to_tool(Invokers.LITELLM)
            if tool:
                tools.append(tool)
        for server in prompt.mcp_servers.values():
            tool = server.to_tool(Invokers.LITELLM)
            if tool:
                tools.append(tool)
        return tools


    def _build_usage(self, usage_dict: dict, response_obj: Any, model: str) -> dict:
        import litellm
        
        # 1. Extract total cost directly from the response object
        usage_dict["response_cost"] = getattr(response_obj, "_hidden_params", {}).get("response_cost", 0.0)
        
        p_tokens = usage_dict.get('prompt_tokens', 0)
        c_tokens = usage_dict.get('completion_tokens', 0)
        
        # 2. Fetch granular dollar costs
        try:
            # litellm returns a tuple: (prompt_cost, completion_cost)
            p_cost, c_cost = litellm.cost_per_token(model=model, prompt_tokens=p_tokens, completion_tokens=c_tokens)
            usage_dict["prompt_cost"] = p_cost
            usage_dict["completion_cost"] = c_cost
        except Exception:
            usage_dict["prompt_cost"] = 0.0
            usage_dict["completion_cost"] = 0.0

        # 3. Fetch specific token rates for the record
        try:
            model_info = litellm.get_model_info(model)
            usage_dict["input_cost_per_token"] = model_info.get("input_cost_per_token", 0.0)
            usage_dict["output_cost_per_token"] = model_info.get("output_cost_per_token", 0.0)
            usage_dict["cache_read_input_token_cost"] = model_info.get("cache_read_input_token_cost", 0.0)
        except Exception:
            usage_dict["input_cost_per_token"] = 0.0
            usage_dict["output_cost_per_token"] = 0.0
            usage_dict["cache_read_input_token_cost"] = 0.0
            
        return usage_dict

    

    def _call_chat_api(
        self,
        dialog: Dialog,
        model: str,
        payload_args: Dict[str, Any],
        parser_args: Dict[str, Any],
        responder: str,
        metadata: Dict[str, Any],
        stream_handler: BaseStreamHandler = None, 
    ) -> Message:
        prompt = dialog.top_prompt
        tools = self._build_tools(prompt)
        call_args = dict(payload_args)

        streaming = stream_handler is not None

        if prompt.format is not None:
            if hasattr(prompt.format, "model_json_schema"): # Pydantic model
                call_args['response_format'] = prompt.format
            else: # dict/schema
                call_args['response_format'] = {"type": "json_object"}

        # if is_reasoning:
        #     call_args['temperature'] = call_args.get('temperature', 1)

        completion = completion_api(
            model=model,
            messages=self._convert_dialog(dialog),
            tools=tools if tools else None,
            **call_args,
        )

        if streaming:
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        stream_handler.handle_chunk(delta.content, chunk)
            completion = stream_chunk_builder(chunks, messages=self._convert_dialog(dialog))
        return self._parse_chat_response(completion, prompt, model, call_args, parser_args, responder, metadata)

    def _parse_chat_response(
        self, completion, prompt, model, call_args, parser_args, responder, metadata
    ) -> Message:

        choice = completion.choices[0]
        usage = {}
        if getattr(completion, "usage", None):
            usage = completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else dict(completion.usage)
        usage = self._build_usage(usage, completion, model)

        if choice.finish_reason == 'tool_calls':
            role = Roles.TOOL_CALL
            logprobs = None
            parsed = None
            errors: List[Exception] = []
            function_calls = [
                FunctionCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in choice.message.tool_calls
            ]
            content = 'Tool calls:\n\n' + '\n'.join(
                [
                    f'{idx}. {tool_call.function.name}: {tool_call.function.arguments}'
                    for idx, tool_call in enumerate(choice.message.tool_calls)
                ]
            )
        else:
            role = Roles.ASSISTANT
            errors = []
            function_calls = []
            content = choice.message.content

            if prompt.format is None:
                raw_logprobs = choice.logprobs.content if choice.logprobs is not None else None
                if raw_logprobs is not None:
                    converted = []
                    for logprob in raw_logprobs:
                        payload = logprob.model_dump() if hasattr(logprob, "model_dump") else logprob
                        converted.append(TokenLogprob.model_validate(payload))
                    logprobs = converted
                else:
                    logprobs = None
                try:
                    parsed = prompt.parse(content, **parser_args) if prompt.parser is not None else None
                except Exception as exc:
                    errors.append(exc)
                    parsed = {'raw': content}
            else:
                try:
                    parsed = json.loads(content)
                except Exception as exc:
                    errors.append(exc)
                    parsed = {'raw': content}
                logprobs = None

            if 'response_format' in call_args and prompt.format is not None:
                call_args['response_format'] = prompt.format.model_json_schema()

        return Message(
            role=role,
            raw_response=completion,
            name=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs or [],
            model=model,
            model_args=call_args,
            usage=usage,
            parsed=parsed or {},
            metadata=metadata,
            execution_errors=errors,
            api_type=APITypes.COMPLETION,
        )

    def _call_response_api(
        self,
        dialog: Dialog,
        model: str,
        payload_args: Dict[str, Any],
        parser_args: Dict[str, Any],
        responder: str,
        metadata: Dict[str, Any],
        stream_handler: BaseStreamHandler = None,
    ) -> Message:
        prompt = dialog.top_prompt
        streaming = stream_handler is not None
        if prompt.format is not None:
            raise ValueError("Response API does not support structured output. Remove 'format' or use the completion API.")

        tools = self._build_tools(prompt)
        if prompt.allow_web_search:
            tools.append({"type": "web_search_preview"})
        if prompt.computer_use_config:
            cfg = prompt.computer_use_config
            tools.append(
                {
                    "type": "computer_use_preview",
                    "display_width": cfg.get("display_width", 1280),
                    "display_height": cfg.get("display_height", 800),
                    "environment": cfg.get("environment", "browser"),
                }
            )

        call_args = dict(payload_args)

        max_output_tokens = call_args.pop('max_output_tokens', call_args.pop('max_completion_tokens', 32000))
        truncation = call_args.pop('truncation', 'auto')
        tool_choice = call_args.pop('tool_choice', 'auto')

        response = responses_api(
            model=model,
            input=self._convert_dialog(dialog),
            tools=tools if tools else None,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            truncation=truncation,
            **call_args,
        )

        if streaming:
            full_response = None
            for event in response:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    stream_handler.handle_chunk(delta_text, event)
                elif event_type == "response.completed":
                    full_response = getattr(event, "response", None)

            if full_response is None:
                 raise ValueError("Streaming finished but no 'response.completed' payload was captured.")
            response = full_response

        return self._parse_responses_api_response(response, prompt, model, call_args, parser_args, responder, metadata)

    def _parse_responses_api_response(
        self, response, prompt, model, call_args, parser_args, responder, metadata
    ) -> Message:

        usage = {}
        if getattr(response, "usage", None):
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
        usage = self._build_usage(usage, response, model)
        outputs = getattr(response, "output", []) or []
        function_calls: List[FunctionCall] = []

        for item in outputs:
            if getattr(item, "type", None) == "function_call":
                arguments = getattr(item, "arguments", "{}")
                try:
                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = {}
                function_calls.append(
                    FunctionCall(
                        id=getattr(item, "call_id", getattr(item, "id", "tool_call")),
                        name=getattr(item, "name", "function"),
                        arguments=parsed_args,
                    )
                )

        logprobs = None
        errors: List[Exception] = []
        if function_calls:
            role = Roles.TOOL_CALL
            parsed = None
            content = 'Tool calls:\n\n' + '\n'.join(
                [f'{idx}. {call.name}: {json.dumps(call.arguments)}' for idx, call in enumerate(function_calls)]
            )
        else:
            role = Roles.ASSISTANT
            content = getattr(response, "output_text", None)
            if not content:
                text_chunks = []
                for item in outputs:
                    if getattr(item, "type", None) == "output_text":
                        chunk = getattr(item, "text", None)
                        if chunk:
                            text_chunks.append(chunk)
                content = '\n'.join(text_chunks).strip()
            try:
                parsed = prompt.parse(content, **parser_args) if prompt.parser is not None else None
            except Exception as exc:
                errors.append(exc)
                parsed = {'raw': content}

        metadata_payload = dict(metadata)
        reasoning = getattr(response, "reasoning", None)
        if reasoning is not None:
            try:
                metadata_payload['reasoning'] = reasoning.model_dump_json()
            except Exception:
                metadata_payload['reasoning'] = str(reasoning)
        metadata_payload.setdefault('api_type', APITypes.RESPONSE.value)

        message = Message(
            role=role,
            raw_response=response,
            name=responder,
            function_calls=function_calls,
            content=content,
            logprobs=logprobs or [],
            model=model,
            model_args=call_args,
            usage=usage,
            parsed=parsed or {},
            metadata=metadata_payload,
            execution_errors=errors,
            api_type=APITypes.RESPONSE,
        )
        return message

    def call(
        self,
        dialog: Dialog,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        metadata: Optional[Dict[str, Any]] = None,
        api_type: APITypes = APITypes.COMPLETION,
        stream_handler: BaseStreamHandler = None,
    ) -> Message:
        """
        Call the API and return the message from the LLM after parsing.

        Example usage:

        - Non-streaming:
        ```python
        message = agent.call(dialog)
        ```

        - Streaming:
        ```python
        class MyStreamHandler(BaseStreamHandler):
            def handle_chunk(self, chunk_content: str, chunk_response: Any):
                print(chunk_content)
        message = agent.call(dialog, stream_handler=MyStreamHandler()) 
        ```
        """
        payload_args = dict(model_args) if model_args else {}
        parser_args = dict(parser_args) if parser_args else {}
        metadata_payload = dict(metadata) if metadata else {}

        payload_args["drop_params"] = True
        payload_args["stream"] = stream_handler is not None
        if stream_handler is not None:
            payload_args["stream_options"] = {"include_usage": True}

        if api_type == APITypes.RESPONSE:
            call_func = self._call_response_api
        else:
            call_func = self._call_chat_api
        return call_func(
            dialog=dialog,
            model=model,
            payload_args=payload_args,
            parser_args=parser_args,
            responder=responder,
            metadata=metadata_payload,
            stream_handler=stream_handler,
        )
