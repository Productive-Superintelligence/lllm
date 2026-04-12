from __future__ import annotations

import hashlib
import inspect
import json
import re
from functools import cached_property
import string
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from lllm.core.const import (
    APITypes,
    InvokeCost,
    InvokeResult,
    Invokers,
    Modalities,
    ParseError,
    FunctionCall,
    Roles,
)
from lllm.core.dialog import Message
from lllm.core.runtime import Runtime, get_default_runtime
import lllm.utils as U
import logging as _logging

from abc import ABC, abstractmethod

_logger = _logging.getLogger(__name__)



class AgentException(Exception):
    """Raised inside the agent loop when parsing or validation fails."""

    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class AgentCallSession(BaseModel):
    """
    Tracking the state of the agent call, including the exceptions, interrupts, and LLM recalls.

    It can be helpful for debugging, analysis, and writing custom handlers. Like error-wise hints, interrupt-wise hints, etc.
    """
    agent_name: str
    max_exception_retry: int
    max_interrupt_steps: int
    max_llm_recall: int

    exception_retries: Dict[str, List[Exception]] = Field(default_factory=dict) # records the exceptions during the agent call at each interrupt step
    interrupts: Dict[str, List[FunctionCall]] = Field(default_factory=dict) # records the function calls during the agent call at each interrupt step
    llm_recalls: Dict[str, List[Exception]] = Field(default_factory=dict) # records the LLM recalls during the agent call at each interrupt step
    invoke_traces: Dict[str, List[InvokeResult]] = Field(default_factory=dict) # all InvokeResult traces at each interrupt step (including failed retries and the final successful one)

    delivery: Optional[Message] = None # the delivered message to the user after *successful* call
    state: Literal["initial", "exception", "interrupt", "llm_recall", "success", "failure"] = "initial"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def exception_retries_count(self) -> int:
        return sum(len(exceptions) for exceptions in self.exception_retries.values())

    @property
    def llm_recalls_count(self) -> int:
        return sum(len(llm_recalls) for llm_recalls in self.llm_recalls.values())

    @property
    def reach_max_exception_retry(self) -> bool:
        return self.exception_retries_count >= self.max_exception_retry

    @property
    def reach_max_llm_recall(self) -> bool:
        return self.llm_recalls_count >= self.max_llm_recall

    @property
    def reach_max_interrupt_steps(self) -> bool:
        return len(self.interrupts) >= self.max_interrupt_steps

    def new_invoke_trace(self, invoke_result: InvokeResult, interrupt_step: int) -> None:
        if interrupt_step not in self.invoke_traces:
            self.invoke_traces[interrupt_step] = []
        self.invoke_traces[interrupt_step].append(invoke_result)

    def exception(self, exception: Exception, interrupt_step: int) -> None:
        _logger.warning(
            "[%s] Handling exception (retry %d/%d): %s",
            self.agent_name, self.exception_retries_count, self.max_exception_retry, exception,
        )
        if interrupt_step not in self.exception_retries:
            self.exception_retries[interrupt_step] = []
        self.exception_retries[interrupt_step].append(exception)
        self.state = "exception"

    def interrupt(self, function_calls: List[FunctionCall], interrupt_step: int) -> None:
        fc_names = [fc.name for fc in function_calls]
        _logger.debug(
            "[%s] Tool calls at step %d/%d: %s",
            self.agent_name, interrupt_step + 1, self.max_interrupt_steps, fc_names,
        )
        for fc in function_calls:
            _logger.debug(
                "[%s] Calling %s with args: %s", self.agent_name, fc.name, fc.arguments,
            )
        if interrupt_step not in self.interrupts:
            self.interrupts[interrupt_step] = []
        self.interrupts[interrupt_step].append(function_calls)
        self.state = "interrupt"

    def llm_recall(self, exception: Exception, interrupt_step: int) -> None:
        _logger.warning(
            "[%s] LLM recall at step %d: %s", self.agent_name, interrupt_step, exception,
        )
        if interrupt_step not in self.llm_recalls:
            self.llm_recalls[interrupt_step] = []
        self.llm_recalls[interrupt_step].append(exception)
        self.state = "llm_recall"

    def success(self, message: Message) -> None:
        _logger.debug(
            "[%s] Agent call succeeded after %d interrupt steps",
            self.agent_name, len(self.interrupts),
        )
        self.state = "success"
        self.delivery = message

    def failure(self) -> None:
        _logger.error("[%s] Agent call failed", self.agent_name)
        self.state = "failure"

    @property
    def cost(self) -> InvokeCost:
        _cost = InvokeCost()
        for invoke_trace in self.invoke_traces.values():
            for invoke_result in invoke_trace:
                _cost += invoke_result.cost
        return _cost

AgentCallSession.model_rebuild()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


# Type mapping for the @tool decorator's auto-inference
_PY_TYPE_TO_JSON: Dict[type, str] = {
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

class Function(BaseModel):
    """
    Declarative description of a callable tool.

    The *schema* (name, description, properties, required) describes the tool
    to the LLM.  The *implementation* is attached separately via
    :meth:`link_function` or by using the :func:`tool` decorator which does
    both in one step.
    """

    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)
    additional_properties: bool = False
    strict: bool = True

    # Implementation (attached at runtime or via decorator)
    function: Optional[Callable] = None
    processor: Callable = _default_function_call_processor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- Linking ----------------------------------------------------------

    def link_function(self, fn: Any) -> None:
        """Attach the Python callable that backs this tool."""
        if not callable(fn):
            raise TypeError(f"Expected a callable for function '{self.name}', got {type(fn)!r}")
        self.function = fn

    @property
    def linked(self) -> bool:
        return self.function is not None

    # -- Execution --------------------------------------------------------

    def __call__(self, function_call: FunctionCall) -> FunctionCall:
        assert self.function is not None, f"Function '{self.name}' not linked"
        try:
            result = self.function(**function_call.arguments)
        except Exception as e:
            function_call.error_message = str(e)
            function_call.result_str = f'Error: {e}'
            return function_call
        function_call.result = result
        function_call.result_str = self.processor(result, function_call)
        return function_call

    def to_tool(self, invoker: Invokers = Invokers.LITELLM) -> Optional[Dict[str, Any]]:
        # This logic might be moved to invoker specific implementations later
        if invoker == Invokers.LITELLM:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": self.properties,
                        "required": self.required,
                        "additionalProperties": self.additional_properties,
                    },
                    "strict": self.strict
                }
            }
        raise NotImplementedError(f"Invoker {invoker} not supported for tool conversion yet")

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prop_desc: Optional[Dict[str, str]] = None,
        strict: bool = True,
        processor: Callable = _default_function_call_processor,
    ) -> Function:
        """
        Build a :class:`Function` by inspecting *fn*'s signature and
        docstring.  Type hints are converted to JSON Schema types.

        Parameters without a type annotation default to ``"string"``.
        Parameters whose names end with ``*`` in the docstring (or that
        lack defaults) are treated as required.

        For example:
        ```python
        @tool(
            description="Get the current weather in a given location"
            prop_desc={
                "location": "The city and state, e.g. San Francisco, CA",
                "unit": "The unit of temperature, e.g. celsius, fahrenheit",
            }
        )
        def get_weather(location: str, unit: str = "celsius") -> str:
            ... # whatever you want to return, be sure to return a string at the end
        ```
        """
        func_name = name or fn.__name__
        func_desc = description or (inspect.getdoc(fn) or func_name)
        sig = inspect.signature(fn)
        hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

        prop_desc: Dict[str, str] = prop_desc or {}
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            py_type = hints.get(param_name, str)
            # Handle Optional[X] → extract X
            origin = getattr(py_type, "__origin__", None)
            if origin is Union:
                args = [a for a in py_type.__args__ if a is not type(None)]
                py_type = args[0] if args else str

            json_type = _PY_TYPE_TO_JSON.get(py_type, "string")
            prop: Dict[str, Any] = {"type": json_type}

            # Use default as example in description
            if param_name in prop_desc:
                prop["description"] = prop_desc[param_name]
            if param.default is not inspect.Parameter.empty:
                if "description" in prop:
                    prop["description"] += f" (default: {param.default!r})"
                else:
                    prop["description"] = f"(default: {param.default!r})"
            else:
                required.append(param_name)

            properties[param_name] = prop

        return cls(
            name=func_name,
            description=func_desc,
            properties=properties,
            required=required,
            strict=strict,
            function=fn,
            processor=processor,
        )

def tool(
    description: Optional[str] = None,
    prop_desc: Optional[Dict[str, str]] = None, # description of the properties, if not provided, it will use default
    *,
    name: Optional[str] = None,
    strict: bool = True,
    processor: Callable = _default_function_call_processor,
) -> Callable[[Callable], Function]:
    """
    Decorator that turns a plain Python function into a :class:`Function`.

    Usage::

        @tool(description="Get current weather for a city")
        def get_weather(location: str, units: str = "celsius") -> str:
            return "Sunny, 22°C"

        # get_weather is now a Function instance with the callable already linked.
        prompt = Prompt(path="bot", prompt="...", function_list=[get_weather])
    """

    def decorator(fn: Callable) -> Function:
        return Function.from_callable(
            fn,
            name=name,
            description=description,
            prop_desc=prop_desc,
            strict=strict,
            processor=processor,
        )

    return decorator


class MCP(BaseModel):
    server_label: str
    server_url: str
    require_approval: Literal["never", "manual", "auto"] = "never"
    allowed_tools: Optional[List[str]] = None

    @field_validator("require_approval")
    @classmethod
    def _validate_approval(cls, value: str) -> str:
        allowed = {"never", "manual", "auto"}
        if value not in allowed:
            raise ValueError(f"require_approval must be one of {allowed}, got {value}")
        return value

    def to_tool(self, invoker: Invokers = Invokers.LITELLM) -> Optional[Dict[str, Any]]:
        if invoker == Invokers.LITELLM:
            tool: Dict[str, Any] = {
                "type": "mcp",
                "server_label": self.server_label,
                "server_url": self.server_url,
                "require_approval": self.require_approval,
            }
            if self.allowed_tools:
                tool["allowed_tools"] = self.allowed_tools
            return tool
        return None




# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


class BaseParser(ABC):
    """
    Base class for all parsers.

    A parser should have a parse method that takes the str content and returns a dictionary of the parsed result.
    """
    @abstractmethod
    def parse(self, content: str, **runtime_args: Any) -> Dict[str, Any]:
        pass


class DefaultTagParser(BaseParser, BaseModel):
    """
    Default tagged language parser for the prompt.

    It will find all xml blocks, md blocks, and signal tags in the message,
    and return a dictionary of the blocks and signal tags. For example, if the message is:
    ```
    <tag1>content1</tag1>
    <tag2>content2</tag2>
    ...
    ```tag3 ... ```
    ```tag4 ... ```
    <STOP_TAG>
    ```
    The xml_tags should be ['tag1', 'tag2'], the md_tags should be ['tag3', 'tag4'],
    and the signal_tags should be ['STOP_TAG']. And the parser will return:
    ```
    {
        'raw': ...,
        'xml_tags': {
            'tag1': ['content1'],
            'tag2': ['content2']
        },
        'md_tags': {
            'tag3': ['content3'],
            'tag4': ['content4']
        },
        'signal_tags': {
            'STOP_TAG': True,
        }
    }
    ```

    Advanced usage:
    You may inherit this class to build stronger parser with more parsing or validation logic, like:
    ```python
    class CustomTagParser(DefaultTagParser):
        def parse(self, content: str, **runtime_args: Any) -> Dict[str, Any]:
            parsed = super().parse(content, **runtime_args)
            ... # do more parsing or validation logic, like if your output is a graph, you may detect cycle
            return parsed
    ```
    """

    # Build-in tag extraction hints for the default parser and validator
    xml_tags: List[str] = Field(default_factory=list)
    md_tags: List[str] = Field(default_factory=list)
    signal_tags: List[str] = Field(default_factory=list)
    required_xml_tags: List[str] = Field(default_factory=list)
    required_md_tags: List[str] = Field(default_factory=list)

    # custom parser args
    parser_args: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, content: str, **runtime_args: Any) -> Dict[str, Any]:
        """
        Parse raw LLM output into a structured dict.

        Raises :class:`~lllm.core.const.ParseError` on failure so the agent
        loop can route to the exception handler.
        """
        xml_tag_blocks = {}
        md_tag_blocks = {}
        errors = []
        for tag in self.xml_tags:
            matches = U.find_xml_blocks(content, tag)
            if len(matches) == 0:
                errors.append(f"No {tag} tags found, it should be provided as <{tag}>...</{tag}>")
            xml_tag_blocks[tag] = matches
        for tag in self.md_tags:
            matches = U.find_md_blocks(content, tag)
            if len(matches) == 0:
                errors.append(f"No {tag} tags found, it should be provided as ```{tag} ... ```")
            md_tag_blocks[tag] = matches
        parsed = {
            'raw': content,
            'xml_tags': xml_tag_blocks,
            'md_tags': md_tag_blocks,
            'signal_tags': {k: f'<{k}>' in content for k in self.signal_tags}
        }
        xml_blocks = parsed.get('xml_tags', {})
        md_blocks = parsed.get('md_tags', {})
        for tag in self.required_xml_tags:
            if tag not in xml_blocks:
                errors.append(f"Missing required XML tag: <{tag}>")
        for tag in self.required_md_tags:
            if tag not in md_blocks:
                errors.append(f"Missing required markdown section: {tag}")
        if len(errors) > 0:
            error_text = "\n".join(errors)
            raise ParseError(f"Parsing errors:\n{error_text}")
        return parsed


class BaseRenderer(ABC):
    """
    Base class for all renderers.

    By default we just use python string formatter for simplicity,
    but you can use more complex renderers like jinja2, or other template engines.
    """
    @abstractmethod
    def render(self, prompt: str, **kwargs: Any) -> str:
        pass


class StringFormatterRenderer(BaseRenderer):
    """
    Default python string formatter renderer for the prompt.
    """
    def render(self, prompt: str, **kwargs: Any) -> str:
        if not kwargs:
            return prompt
        return prompt.format(**kwargs)


class BaseHandler(ABC):
    """
    Base class for all handlers.

    You may use call_state to build more complex handlers, like rule-based,
    event-driven, or even an agentic handler like bug fixing agent.
    """
    @abstractmethod
    def on_exception(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        pass

    @abstractmethod
    def on_interrupt(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        pass

    @abstractmethod
    def on_interrupt_final(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        pass



_DEFAULT_EXCEPTION_MSG = "Error: {error_message}. Please fix."
_DEFAULT_INTERRUPT_MSG = "{call_results}"
_DEFAULT_INTERRUPT_FINAL_MSG = "You are reaching the limit of tool calls. Provide the final response."

class DefaultSimpleHandler(BaseHandler, BaseModel):
    """
    A simple handler for the prompt.

    User may override the exception message, interrupt message, and interrupt final message to build a more complex handler.
    """
    exception_msg: str = _DEFAULT_EXCEPTION_MSG
    interrupt_msg: str = _DEFAULT_INTERRUPT_MSG
    interrupt_final_msg: str = _DEFAULT_INTERRUPT_FINAL_MSG


    def on_exception(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        return self._resolve_handler(
            prompt,
            self.exception_msg,
            suffix="exception",
            inherit_tools=True,
        )

    def on_interrupt(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        return self._resolve_handler(
            prompt,
            self.interrupt_msg,
            suffix="interrupt",
            inherit_tools=True,
        )

    def on_interrupt_final(self, prompt: Prompt, session: AgentCallSession) -> Prompt:
        return self._resolve_handler(
            prompt,
            self.interrupt_final_msg,
            suffix="interrupt_final",
            inherit_tools=False,
        )

    def _resolve_handler(
        self,
        prompt: Prompt,
        handler: Union[str, Prompt],
        suffix: str,
        inherit_tools: bool,
    ) -> Prompt:
        if isinstance(handler, Prompt):
            return handler
        return prompt.extend(
            path=f"__{prompt.path}_{suffix}",
            prompt=handler,
            function_list=prompt.function_list if inherit_tools else [],
            mcp_servers_list=prompt.mcp_servers_list if inherit_tools else [],
            addon_args=prompt.addon_args if inherit_tools else {},
        )



class Prompt(BaseModel):
    """
    A Prompt is a complete behaviour definition for one agent turn.

    It bundles four concerns:

    1. **Template** — the text to send to the LLM, with ``{variable}``
       placeholders rendered via ``str.format`` (or a custom ``renderer``).
    2. **Output contract** — an :class:`OutputSpec` describing how to parse
       and validate the LLM's response.
    3. **Tools** — the :class:`Function` and :class:`MCP` objects available
       during this turn.
    4. **Handlers** — template strings (or full Prompts) that define how to
       recover from exceptions and how to feed tool results back.

    Provider-specific features (web search, computer use, citations, …) live
    in the generic ``capabilities`` dict so that new features never require
    schema changes on Prompt.

    Notes
    -----
    The ``__call__`` method uses ``str.format`` by default, so literal braces
    in the template must be doubled: ``{{`` and ``}}``.
    """
    path: str
    prompt: str
    metadata: Dict[str, Any] = Field(default_factory=dict) # record additional info, like version, etc.

    # -- Output contract --------------------------------------------------
    parser: Optional[BaseParser] = None
    format: Optional[Any] = None # Structured output (Pydantic model class or JSON schema dict)

    # -- Tools ------------------------------------------------------------
    function_list: List[Function] = Field(default_factory=list)
    mcp_servers_list: List[MCP] = Field(default_factory=list)

    # Provider-specific capabilities or args (like allow_web_search, computer_use_config, etc.)
    addon_args: Dict[str, Any] = Field(default_factory=dict)

    # -- Handlers ---------------------------------------------------------
    handler: BaseHandler = Field(default_factory=DefaultSimpleHandler)

    # -- Rendering --------------------------------------------------------
    renderer: BaseRenderer = Field(default_factory=StringFormatterRenderer)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- Internal (populated in model_post_init) --------------------------
    _functions: Dict[str, Function] = PrivateAttr(default_factory=dict)
    _mcp_servers: Dict[str, MCP] = PrivateAttr(default_factory=dict)
    _template_vars: set = PrivateAttr(default_factory=set)

    def model_post_init(self, __context):
        self._functions = {f.name: f for f in self.function_list}
        self._mcp_servers = {m.server_label: m for m in self.mcp_servers_list}

        _parser = string.Formatter()
        self._template_vars = {
            field_name.split('.')[0].split('[')[0]
            for _, field_name, _, _ in _parser.parse(self.prompt)
            if field_name is not None
        }

    @property
    def functions(self) -> Dict[str, Function]:
        return self._functions

    @property
    def mcp_servers(self) -> Dict[str, MCP]:
        return self._mcp_servers


    @property
    def template_vars(self) -> set[str]:
        """Variable names required by this template (e.g. {topic} → 'topic')."""
        return self._template_vars

    def validate_args(self, prompt_args: dict[str, Any]) -> list[str]:
        """Return list of missing required template variables for the prompt."""
        return [v for v in self.template_vars if v not in prompt_args]


    # -- Rendering --------------------------------------------------------

    def __call__(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        if not self.template_vars:  # empty set is falsy
            return self.prompt
        missing_args = self.validate_args(kwargs)
        if missing_args:
            raise ValueError(f"Missing required template variables: {missing_args} for prompt {self.path}, please provide: {self.template_vars}")
        return self.renderer.render(self.prompt, **kwargs)


    def parse(self, content: str, **runtime_args: Any) -> Dict[str, Any]:
        if self.parser is not None:
            parsed = self.parser.parse(content, **runtime_args)
            if 'raw' not in parsed:
                parsed['raw'] = content
            return parsed
        return {
            'raw': content,
        }

    # -- Tool management --------------------------------------------------

    def link_function(self, name: str, fn: Callable) -> None:
        """Attach a Python callable to an already-declared Function by name."""
        if name not in self.functions:
            raise KeyError(
                f"Function '{name}' not declared on prompt '{self.path}'. "
                f"Available: {sorted(self.functions)}"
            )
        self.functions[name].link_function(fn)

    def get_function(self, name: str) -> Function:
        """Retrieve a declared Function by name, with a clear error."""
        if name not in self.functions:
            raise KeyError(
                f"Function '{name}' not found on prompt '{self.path}'. "
                f"Available: {sorted(self.functions)}"
            )
        return self.functions[name]

    def register_mcp_server(self, server: MCP) -> None:
        self.mcp_servers[server.server_label] = server


    # -- Handler management -----------------------------------------------

    def on_exception(self, session: AgentCallSession) -> Prompt:
        return self.handler.on_exception(self, session)

    def on_interrupt(self, session: AgentCallSession) -> Prompt:
        return self.handler.on_interrupt(self, session)

    def on_interrupt_final(self, session: AgentCallSession) -> Prompt:
        return self.handler.on_interrupt_final(self, session)

    # -- Capability accessors (convenience, read-only) --------------------

    @property
    def allow_web_search(self) -> bool:
        return bool(self.addon_args.get("web_search", False))

    @property
    def computer_use_config(self) -> Dict[str, Any]:
        return self.addon_args.get("computer_use", {})

    # -- Composition ------------------------------------------------------

    def extend(self, **overrides: Any) -> Prompt:
        """
        Create a new Prompt inheriting all fields, with *overrides* applied.

        A new ``path`` is required — prompts that share a path would collide
        in the registry::

            child = parent.extend(
                path="child/analysis",
                prompt="More specific: {task}",
                output=OutputSpec(parser=strict_parser),
            )
        """
        if "path" not in overrides:
            raise ValueError("extend() requires a new 'path'")
        # Build from current field values directly, not via serialization
        current = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name not in ("_functions", "_mcp_servers")
        }
        current.update(overrides)
        return Prompt(**current)

    # -- Metadata for logging / tracking ----------------------------------

    def info_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot suitable for experiment tracking.
        """
        return {
            "path": self.path,
            "prompt_hash": hashlib.sha256(self.prompt.encode()).hexdigest()[:12],
            "metadata": self.metadata,
            "functions": [f.name for f in self.function_list],
            "mcp_servers": [m.server_label for m in self.mcp_servers_list],
            "addon_args": self.addon_args,
            "has_parser": self.parser is not None,
            "has_format": self.format is not None,
        }

# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def register_prompt(prompt: Prompt, overwrite: bool = True) -> None:
    """Register a prompt into the default runtime."""
    get_default_runtime().register_prompt(prompt, overwrite)