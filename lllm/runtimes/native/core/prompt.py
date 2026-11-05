from __future__ import annotations

import copy
import hashlib
import inspect
import logging as _logging
import string
import sys
import types
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictStr,
    field_validator,
)

from ....protocol._validation import (
    copy_boundary_value,
    mapping_field_value,
    metadata_field_value,
    optional_mapping_value,
)
from .. import utils as U
from .const import (
    FunctionCall,
    InvokeCost,
    InvokeResult,
    Invokers,
    ParseError,
    validate_tool_name,
)
from .dialog import Message
from .runtime import Runtime, get_default_runtime

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

    exception_retries: Dict[str, List[Exception]] = Field(
        default_factory=dict
    )  # records the exceptions during the agent call at each interrupt step
    interrupts: Dict[str, List[FunctionCall]] = Field(
        default_factory=dict
    )  # records the function calls during the agent call at each interrupt step
    llm_recalls: Dict[str, List[Exception]] = Field(
        default_factory=dict
    )  # records the LLM recalls during the agent call at each interrupt step
    invoke_traces: Dict[str, List[InvokeResult]] = Field(
        default_factory=dict
    )  # all InvokeResult traces at each interrupt step (including failed retries and the final successful one)

    delivery: Optional[Message] = (
        None  # the delivered message to the user after *successful* call
    )
    state: Literal[
        "initial", "exception", "interrupt", "llm_recall", "success", "failure"
    ] = "initial"

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

    def new_invoke_trace(
        self, invoke_result: InvokeResult, interrupt_step: int
    ) -> None:
        if interrupt_step not in self.invoke_traces:
            self.invoke_traces[interrupt_step] = []
        self.invoke_traces[interrupt_step].append(invoke_result)

    def exception(self, exception: Exception, interrupt_step: int) -> None:
        _logger.warning(
            "[%s] Handling exception (retry %d/%d): %s",
            self.agent_name,
            self.exception_retries_count,
            self.max_exception_retry,
            exception,
        )
        if interrupt_step not in self.exception_retries:
            self.exception_retries[interrupt_step] = []
        self.exception_retries[interrupt_step].append(exception)
        self.state = "exception"

    def interrupt(
        self, function_calls: List[FunctionCall], interrupt_step: int
    ) -> None:
        fc_names = [fc.name for fc in function_calls]
        _logger.debug(
            "[%s] Tool calls at step %d/%d: %s",
            self.agent_name,
            interrupt_step + 1,
            self.max_interrupt_steps,
            fc_names,
        )
        for fc in function_calls:
            _logger.debug(
                "[%s] Calling %s with args: %s",
                self.agent_name,
                fc.name,
                fc.arguments,
            )
        if interrupt_step not in self.interrupts:
            self.interrupts[interrupt_step] = []
        self.interrupts[interrupt_step].append(function_calls)
        self.state = "interrupt"

    def llm_recall(self, exception: Exception, interrupt_step: int) -> None:
        _logger.warning(
            "[%s] LLM recall at step %d: %s",
            self.agent_name,
            interrupt_step,
            exception,
        )
        if interrupt_step not in self.llm_recalls:
            self.llm_recalls[interrupt_step] = []
        self.llm_recalls[interrupt_step].append(exception)
        self.state = "llm_recall"

    def success(self, message: Message) -> None:
        _logger.debug(
            "[%s] Agent call succeeded after %d interrupt steps",
            self.agent_name,
            len(self.interrupts),
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


def _json_type_for_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _json_type_for_annotation(args[0]) if args else "string"
    if origin is not None:
        return _PY_TYPE_TO_JSON.get(origin, "string")
    return _PY_TYPE_TO_JSON.get(annotation, "string")


def _annotation_schema(annotation: Any) -> Dict[str, Any]:
    """JSON schema for a parameter annotation (lists, enums, models, unions)."""

    try:
        from pydantic import TypeAdapter

        schema = _inline_refs(TypeAdapter(annotation).json_schema())
    except Exception:
        return {"type": _json_type_for_annotation(annotation)}
    schema.pop("title", None)
    return schema or {"type": _json_type_for_annotation(annotation)}


def _inline_refs(schema: Any) -> Any:
    """Replace ``{"$ref": "#/$defs/X"}`` with the definition; drop ``$defs``.

    Tool parameters have no place for ``$defs``, so every model is inlined
    where it is used. Within a recursive cycle of models, though, each model
    is spelled out only at its first (shallowest) use; a later reference
    back into the cycle becomes the model's bare type and title
    (``{"type": "object", "title": "X"}``, or ``{}`` for a definition with
    neither). The result grows with the definitions, not with the number
    of paths through a cycle.
    """

    if not isinstance(schema, dict):
        return schema
    defs = schema.get("$defs", {})
    reach = _reachable_defs(defs)
    spelled_out: set[str] = set()
    result: Dict[str, Any] = {}
    # Breadth first, so a model is spelled out at its shallowest use.
    # ``model`` is the innermost definition being inlined.
    queue = deque([(result, "schema", schema, None)])
    while queue:
        parent, key, node, model = queue.popleft()
        while isinstance(node, dict) and (name := _def_name(node, defs)):
            target = defs[name]
            extra = {k: v for k, v in node.items() if k != "$ref"}
            # Back into the cycle being inlined (``name`` leads to ``model``),
            # to a model that is already spelled out.
            if model is not None and model in reach[name] and name in spelled_out:
                bare = {k: target[k] for k in ("type", "title") if k in target}
                node = {**bare, **extra}
                break
            spelled_out.add(name)
            node, model = {**target, **extra}, name
        if isinstance(node, dict):
            parent[key] = {}
            queue.extend(
                (parent[key], k, v, model) for k, v in node.items() if k != "$defs"
            )
        elif isinstance(node, list):
            parent[key] = [None] * len(node)
            queue.extend((parent[key], i, v, model) for i, v in enumerate(node))
        else:
            parent[key] = node
    return result["schema"]


def _def_name(schema: Dict[str, Any], defs: Dict[str, Any]) -> Optional[str]:
    """Name of the definition that *schema* refers to with a ``$ref``."""

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.removeprefix("#/$defs/")
        if name in defs:
            return name
    return None


def _reachable_defs(defs: Dict[str, Any]) -> Dict[str, set[str]]:
    """Map each definition to the definitions its references lead to."""

    def refs(schema: Any) -> set[str]:
        if isinstance(schema, list):
            return set().union(*map(refs, schema))
        if not isinstance(schema, dict):
            return set()
        found = set().union(*map(refs, schema.values()))
        name = _def_name(schema, defs)
        return found | {name} if name else found

    direct = {name: refs(body) for name, body in defs.items()}
    reach: Dict[str, set[str]] = {}
    for name in defs:
        seen: set[str] = set()
        pending = list(direct[name])
        while pending:
            other = pending.pop()
            if other not in seen:
                seen.add(other)
                pending.extend(direct[other])
        reach[name] = seen
    return reach


def _resolve_strict(
    strict: Optional[bool], properties: Dict[str, Any], required: List[str]
) -> bool:
    """Use strict tool schemas only when the provider can accept them.

    OpenAI-style strict mode needs every parameter required and flat schemas;
    an explicit ``strict`` always wins.
    """

    if strict is not None:
        return _bool_value("strict", strict)
    if set(required) != set(properties):
        return False
    return all(_is_flat_schema(schema) for schema in properties.values())


def _is_flat_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    # Strict providers need ``additionalProperties: false`` on every object,
    # so objects (models, dicts, cut-off recursive models) are never flat.
    if schema.get("type") == "object":
        return False
    if any(key in schema for key in ("properties", "anyOf", "oneOf", "allOf", "$ref", "default")):
        return False
    items = schema.get("items")
    return items is None or _is_flat_schema(items)


def _default_function_call_processor(result: Any, function_call: FunctionCall) -> str:
    return (
        f"Return of calling function {function_call.name} "
        f"with arguments {function_call.arguments}:\n---\n{result}\n---\n"
    )


class Function(BaseModel):
    """
    Declarative description of a callable tool.

    The *schema* (name, description, properties, required) describes the tool
    to the LLM.  The implementation is supplied by a package tool resource
    with the same key, usually created by the :func:`tool` decorator.
    """

    name: StrictStr
    description: StrictStr
    properties: Dict[str, Any]
    required: List[StrictStr] = Field(default_factory=list)
    additional_properties: StrictBool = False
    strict: StrictBool = True

    # Implementation (attached at runtime or via decorator)
    function: Optional[Callable] = None
    processor: Callable = _default_function_call_processor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        validate_tool_name(self.name, "function name")
        self.properties = copy_boundary_value(self.properties)
        self.required = copy.deepcopy(self.required)

    @field_validator("properties", mode="before")
    @classmethod
    def _validate_properties(cls, value: Any) -> Any:
        return _mapping_field_value("properties", value)

    # -- Execution --------------------------------------------------------

    def __call__(self, function_call: FunctionCall) -> FunctionCall:
        if self.function is None:
            raise RuntimeError(
                f"Function '{self.name}' has no implementation. "
                "Declare the schema in the prompt and register a matching "
                "package tool implementation with @tool."
            )
        try:
            result = self.function(**function_call.arguments)
        except Exception as e:
            function_call.error_message = str(e)
            function_call.result_str = f"Error: {e}"
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
                        "properties": copy_boundary_value(self.properties),
                        "required": copy.deepcopy(self.required),
                        "additionalProperties": self.additional_properties,
                    },
                    "strict": self.strict,
                },
            }
        raise NotImplementedError(
            f"Invoker {invoker} not supported for tool conversion yet"
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prop_desc: Optional[Mapping[str, str]] = None,
        strict: Optional[bool] = None,
        processor: Callable = _default_function_call_processor,
    ) -> Function:
        """
        Build a :class:`Function` by inspecting *fn*'s signature and
        docstring.  Type hints are converted to JSON Schema types; nested
        models are inlined, and the models of a recursive cycle are spelled
        out once each, with other references back into the cycle described
        by type and title (``{"type": "object", "title": "Node"}``).

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
        func_name = name if name is not None else fn.__name__
        func_desc = description or (inspect.getdoc(fn) or func_name)
        sig = inspect.signature(fn)
        hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}

        property_descriptions = _optional_string_mapping("prop_desc", prop_desc)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            py_type = hints.get(param_name, str)
            prop: Dict[str, Any] = _annotation_schema(py_type)

            # Use default as example in description
            if param_name in property_descriptions:
                prop["description"] = property_descriptions[param_name]
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
            strict=_resolve_strict(strict, properties, required),
            function=fn,
            processor=processor,
        )


def tool(
    description: Optional[str] = None,
    prop_desc: Optional[Mapping[str, str]] = None,
    *,
    name: Optional[str] = None,
    strict: Optional[bool] = None,
    processor: Callable = _default_function_call_processor,
) -> Callable[[Callable], Function]:
    """
    Decorator that turns a plain Python function into a :class:`Function`.

    Usage::

        @tool(description="Get current weather for a city")
        def get_weather(location: str, units: str = "celsius") -> str:
            return "Sunny, 22°C"

        # get_weather is now a Function instance with its implementation.
        prompt = Prompt(path="bot", prompt="...", function_list=[get_weather])
    """

    strict_value = None if strict is None else _bool_value("strict", strict)

    def decorator(fn: Callable) -> Function:
        return Function.from_callable(
            fn,
            name=name,
            description=description,
            prop_desc=prop_desc,
            strict=strict_value,
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
        xml_tags = list(dict.fromkeys([*self.xml_tags, *self.required_xml_tags]))
        md_tags = list(dict.fromkeys([*self.md_tags, *self.required_md_tags]))
        for tag in xml_tags:
            matches = U.find_xml_blocks(content, tag)
            if len(matches) == 0 and tag in self.xml_tags:
                errors.append(
                    f"No {tag} tags found, it should be provided as <{tag}>...</{tag}>"
                )
            xml_tag_blocks[tag] = matches
        for tag in md_tags:
            matches = U.find_md_blocks(content, tag)
            if len(matches) == 0 and tag in self.md_tags:
                errors.append(
                    f"No {tag} tags found, it should be provided as ```{tag} ... ```"
                )
            md_tag_blocks[tag] = matches
        parsed = {
            "raw": content,
            "xml_tags": xml_tag_blocks,
            "md_tags": md_tag_blocks,
            "signal_tags": {k: f"<{k}>" in content for k in self.signal_tags},
        }
        xml_blocks = parsed.get("xml_tags", {})
        md_blocks = parsed.get("md_tags", {})
        for tag in self.required_xml_tags:
            if not xml_blocks.get(tag):
                errors.append(f"Missing required XML tag: <{tag}>")
        for tag in self.required_md_tags:
            if not md_blocks.get(tag):
                errors.append(f"Missing required markdown block: {tag}")
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
        # Always format, so escaped braces ({{ }}) become literal braces even
        # in templates without variables.
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
_DEFAULT_INTERRUPT_FINAL_MSG = (
    "You are reaching the limit of tool calls. Provide the final response."
)


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

    path: StrictStr
    prompt: StrictStr
    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )  # record additional info, like version, etc.

    # -- Output contract --------------------------------------------------
    parser: Optional[Any] = None
    format: Optional[Any] = (
        None  # Structured output (Pydantic model class or JSON schema dict)
    )

    # -- Tools ------------------------------------------------------------
    function_list: List[Union[Function, StrictStr]] = Field(default_factory=list)
    mcp_servers_list: List[MCP] = Field(default_factory=list)
    #: Also offer the agent's own tools (those on its system prompt, including
    #: config ``tools:``, proxy tools, and skills) on this turn. Set False for
    #: a turn that must answer without tools.
    agent_tools: bool = True

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

    def model_post_init(self, __context):
        self._functions = {
            f.name: copy.deepcopy(f)
            for f in self.function_list
            if isinstance(f, Function)
        }
        self._mcp_servers = {
            m.server_label: copy.deepcopy(m) for m in self.mcp_servers_list
        }

        _template_fields(self.prompt)  # a malformed template fails here
        self.metadata = copy_boundary_value(self.metadata)
        self.addon_args = copy_boundary_value(self.addon_args)
        self.function_list = copy.deepcopy(self.function_list)
        self.mcp_servers_list = copy.deepcopy(self.mcp_servers_list)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return metadata_field_value("metadata", value)

    @field_validator("addon_args", mode="before")
    @classmethod
    def _validate_addon_args(cls, value: Any) -> Any:
        return _mapping_field_value("addon_args", value)

    @property
    def functions(self) -> Dict[str, Function]:
        # ``model_copy(update={"function_list": ...})`` does not rebuild
        # private attrs in pydantic v2. Rebuild lazily so copied prompts and
        # resolved tactic refs expose the correct tool map.
        functions = {
            f.name: copy.deepcopy(f)
            for f in self.function_list
            if isinstance(f, Function)
        }
        if set(functions) != set(self._functions) or any(
            self._functions.get(k) != v for k, v in functions.items()
        ):
            self._functions = functions
        return copy.deepcopy(self._functions)

    @property
    def mcp_servers(self) -> Dict[str, MCP]:
        return copy.deepcopy(self._mcp_servers)

    @property
    def template_vars(self) -> set[str]:
        """Variable names required by this template (e.g. {topic} → 'topic')."""
        # Parsed from the current text: ``model_copy(update={"prompt": ...})``
        # skips ``model_post_init``, so a cached set would go stale.
        return _template_fields(self.prompt)

    def validate_args(self, prompt_args: dict[str, Any]) -> list[str]:
        """Return list of missing required template variables for the prompt."""
        return [v for v in self.template_vars if v not in prompt_args]

    # -- Rendering --------------------------------------------------------

    def __call__(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Templates use ``str.format`` syntax, so literal braces are written
        ``{{`` / ``}}``; they are unescaped even when there are no variables.
        """
        if not self.template_vars:
            # No str.format fields: a custom renderer may still use its own
            # syntax (e.g. string.Template), so it receives the variables.
            return self.renderer.render(self.prompt, **kwargs)
        missing_args = self.validate_args(kwargs)
        if missing_args:
            raise ValueError(
                f"Missing required template variables: {missing_args} for prompt {self.path}, please provide: {self.template_vars}"
            )
        return self.renderer.render(self.prompt, **kwargs)

    def parse(self, content: str, **runtime_args: Any) -> Dict[str, Any]:
        if self.parser is not None and _has_static_callable(self.parser, "parse"):
            parsed = self.parser.parse(content, **runtime_args)
        elif callable(self.parser):
            parsed = self.parser(content, **runtime_args)
        else:
            return {
                "raw": content,
            }
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        if "raw" not in parsed:
            parsed["raw"] = content
        return parsed

    # -- Tool management --------------------------------------------------

    def get_function(self, name: str) -> Function:
        """Retrieve a declared Function by name, with a clear error."""
        functions = self.functions
        if name not in functions:
            raise KeyError(
                f"Function '{name}' not found on prompt '{self.path}'. "
                f"Available: {sorted(functions)}"
            )
        return functions[name]

    def register_mcp_server(self, server: MCP) -> None:
        self._mcp_servers[server.server_label] = copy.deepcopy(server)

    def resolve_function_refs(self, runtime: Optional[Runtime] = None) -> "Prompt":
        """
        Resolve tool declarations into executable ``Function``s.

        String entries may be regular tool resources or tactic tool resources.
        Unimplemented ``Function`` objects are declarations that bind to a
        regular tool resource with the same key in the prompt's package.
        Resolution is package-local first when the prompt has a qualified
        resource key.
        """
        runtime = runtime or get_default_runtime()
        needs_resolution = any(
            isinstance(item, str)
            or (isinstance(item, Function) and item.function is None)
            for item in self.function_list
        )
        if not needs_resolution:
            # Still force a lazy rebuild for prompts produced by model_copy.
            self._functions = {
                f.name: f for f in self.function_list if isinstance(f, Function)
            }
            return self

        from .tactic_tool import (
            bind_function_declaration,
            build_prompt_function_ref,
            namespace_from_qualified_key,
        )

        base_namespace = getattr(
            self, "_resource_namespace", None
        ) or namespace_from_qualified_key(getattr(self, "_qualified_key", None))

        resolved: List[Function] = []
        seen: set[str] = set()
        for item in self.function_list:
            if isinstance(item, Function):
                function = bind_function_declaration(
                    item,
                    runtime=runtime,
                    base_namespace=base_namespace,
                )
            elif isinstance(item, str):
                function = build_prompt_function_ref(
                    item,
                    runtime=runtime,
                    base_namespace=base_namespace,
                )
            else:
                raise TypeError(
                    "function_list entries must be Function objects or tool "
                    f"resource strings, got {type(item).__name__}"
                )

            if function.name in seen:
                raise ValueError(
                    f"Duplicate tool name '{function.name}' on prompt '{self.path}'"
                )
            seen.add(function.name)
            resolved.append(function)

        current = {
            name: copy_boundary_value(getattr(self, name))
            if name in {"metadata", "addon_args"}
            else copy.deepcopy(getattr(self, name))
            for name in type(self).model_fields
            if name not in ("_functions", "_mcp_servers")
        }
        current["function_list"] = resolved
        prompt = type(self)(**current)
        qualified_key = getattr(self, "_qualified_key", None)
        if qualified_key is not None:
            prompt._qualified_key = qualified_key  # type: ignore[attr-defined]
        resource_namespace = getattr(self, "_resource_namespace", None)
        if resource_namespace is not None:
            prompt._resource_namespace = resource_namespace  # type: ignore[attr-defined]
        return prompt

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
            name: copy_boundary_value(getattr(self, name))
            if name in {"metadata", "addon_args"}
            else copy.deepcopy(getattr(self, name))
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
            "metadata": copy_boundary_value(self.metadata),
            "functions": [
                f.name if isinstance(f, Function) else f for f in self.function_list
            ],
            "mcp_servers": [m.server_label for m in self.mcp_servers_list],
            "addon_args": copy_boundary_value(self.addon_args),
            "has_parser": self.parser is not None,
            "has_format": self.format is not None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe snapshot (tool callables and parsers are not included).

        Tools are stored as their schemas (implementations are re-bound by
        name when the prompt is resolved against a runtime); a Pydantic
        ``format`` model is stored as its import path and JSON schema.
        """

        data = self.model_dump(
            mode="python",
            exclude={"parser", "handler", "renderer", "function_list", "format"},
        )
        data["function_list"] = [
            item
            if isinstance(item, str)
            else item.model_dump(mode="json", exclude={"function", "processor"})
            for item in self.function_list
        ]
        data["format"] = _format_to_dict(self.format)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        data = dict(data)
        data["format"] = _format_from_dict(data.get("format"))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def _template_fields(template: str) -> set[str]:
    """Top-level ``str.format`` field names in *template* ({a.b} → 'a')."""
    return {
        field_name.split(".")[0].split("[")[0]
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name is not None
    }


def _mapping_field_value(label: str, value: Any) -> Dict[str, Any]:
    return mapping_field_value(label, value)


def _optional_mapping(label: str, value: Any) -> Dict[str, Any]:
    return optional_mapping_value(label, value)


def _optional_string_mapping(label: str, value: Any) -> Dict[str, str]:
    try:
        entries = _optional_mapping(label, value)
    except TypeError as exc:
        if "keys must be strings" in str(exc):
            raise TypeError(f"{label} keys and values must be strings.") from exc
        raise
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
    descriptor_value = getattr(attribute, "__func__", attribute)
    return callable(descriptor_value)


def tactic_as_function(
    tactic: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    parameter_mode: Literal["task", "kwargs"] = "task",
    strict: Optional[bool] = None,
    processor: Callable = _default_function_call_processor,
) -> Function:
    """Expose an LLLM tactic as a native prompt ``Function``.

    The tool's parameters come from the tactic's published input schema:
    one ``task`` parameter holding the whole input (``parameter_mode="task"``)
    or the input model's fields as separate parameters (``"kwargs"``).
    An input without fields (e.g. ``str``) cannot be split into keyword
    arguments, so ``"kwargs"`` falls back to the single ``task`` parameter.
    Nested and recursive models are inlined as in
    :meth:`Function.from_callable`.
    """

    from ...pydantic_ai import tactic_as_tool

    input_schema = _inline_refs(tactic.info().input_schema or {})
    if parameter_mode == "kwargs" and not input_schema.get("properties"):
        parameter_mode = "task"
    callable_tool = tactic_as_tool(
        tactic,
        name=name,
        description=description,
        parameter_mode=parameter_mode,
    )
    if parameter_mode == "kwargs":
        properties = copy.deepcopy(input_schema["properties"])
        required = list(input_schema.get("required", []))
    else:
        properties = {"task": input_schema or {"type": "string"}}
        required = ["task"]
    return Function(
        name=callable_tool.__name__,
        description=inspect.getdoc(callable_tool) or description or "",
        properties=properties,
        required=required,
        strict=_resolve_strict(strict, properties, required),
        function=callable_tool,
        processor=processor,
    )


def register_prompt(prompt: Prompt, overwrite: bool = True) -> None:
    """Register a prompt into the default runtime."""
    get_default_runtime().register_prompt(prompt, overwrite)


def _format_to_dict(value: Any) -> Any:
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return {
            "model": f"{value.__module__}:{value.__qualname__}",
            "schema": value.model_json_schema(),
        }
    return value


def _format_from_dict(value: Any) -> Any:
    if not (isinstance(value, dict) and set(value) == {"model", "schema"}):
        return value
    # Saved data must never import code: use the model class only if its
    # module is already loaded, otherwise keep the stored JSON schema.
    module_name, _, qualname = str(value["model"]).partition(":")
    target: Any = sys.modules.get(module_name)
    if target is None:
        return value["schema"]
    try:
        for part in qualname.split("."):
            target = getattr(target, part)
    except Exception:
        return value["schema"]
    if isinstance(target, type) and hasattr(target, "model_json_schema"):
        return target
    return value["schema"]
