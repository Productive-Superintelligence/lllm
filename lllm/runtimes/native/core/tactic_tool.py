"""
Adapters that expose Tactic resources as agent-callable tools.

This module is intentionally narrow: it does not redefine the tool system.
It resolves tactic resource refs and builds the existing ``Function`` objects
or proxy endpoint callables around them.
"""

from __future__ import annotations

import copy
import inspect
import re
import types
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from .config import resolve_config
from .prompt import Function
from .resource import ResourceNode
from .runtime import Registry, Runtime, _select_runtime

_RESOURCE_SECTIONS = {
    "prompt": "prompts",
    "tool": "tools",
    "proxy": "proxies",
    "tactic": "tactics",
    "config": "configs",
}

_SECTION_TO_RESOURCE_TYPE = {
    "prompt": "prompt",
    "prompts": "prompt",
    "tool": "tool",
    "tools": "tool",
    "proxy": "proxy",
    "proxies": "proxy",
    "tactic": "tactic",
    "tactics": "tactic",
    "config": "config",
    "configs": "config",
}

_JSON_TYPE_TO_PYTHON = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_PY_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass(frozen=True)
class TacticToolMeta:
    """Metadata attached to a Tactic method by ``@tactictool``."""

    name: str
    description: Optional[str] = None
    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None
    config: Any = None


def tactictool(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
    config: Any = None,
):
    """
    Mark a ``Tactic`` method as a callable tool.

    The public tool name is required and lives on the decorator, not the
    tactic class or Python method name. A tactic ref can select a specific
    exposed method with ``#tool_name``.
    """
    if callable(name):
        raise TypeError(
            "@tactictool requires an explicit tool name. "
            "Use @tactictool('name', ...) or @tactictool(name='name', ...)."
        )
    if not name or not str(name).strip():
        raise ValueError(
            "@tactictool requires a non-empty tool name. "
            "Use @tactictool('name', ...) or @tactictool(name='name', ...)."
        )

    def decorate(method: Callable) -> Callable:
        method._lllm_tactic_tool = TacticToolMeta(
            name=str(name),
            description=description,
            input_model=input_model,
            output_model=output_model,
            config=config,
        )
        return method

    return decorate


@dataclass(frozen=True)
class TacticToolSpec:
    """Resolved metadata needed to expose a tactic as a tool."""

    ref: str
    tactic_ref: str
    selector: Optional[str]
    node: ResourceNode
    tactic_cls: Type
    method_name: str
    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str]
    input_model: Optional[Type[BaseModel]]
    output_model: Optional[Type[BaseModel]]
    config: Any = None


def _package_from_namespace(namespace: Optional[str]) -> Optional[str]:
    if not namespace:
        return None
    return namespace.split(".", 1)[0]


def namespace_from_qualified_key(qualified_key: Optional[str]) -> Optional[str]:
    """Return ``pkg.section`` from ``pkg.section:key`` if present."""
    if not qualified_key or ":" not in qualified_key:
        return None
    return qualified_key.split(":", 1)[0]


def sanitize_tool_name(name: str, *, fallback: str = "tactic_tool") -> str:
    """Normalize a resource name into a provider-friendly tool name."""
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "")).strip("_")
    if not value:
        value = fallback
    if value[0].isdigit():
        value = f"tactic_{value}"
    return value


def split_tactic_tool_ref(ref: str) -> tuple[str, Optional[str]]:
    """Split ``pkg.tactics:name#tool`` into tactic ref and optional selector."""
    tactic_ref, sep, selector = ref.partition("#")
    if not tactic_ref:
        raise ValueError(f"Invalid tactic tool ref {ref!r}: missing tactic ref")
    if sep and not selector:
        raise ValueError(
            f"Invalid tactic tool ref {ref!r}: missing tool selector after '#'"
        )
    return tactic_ref, selector or None


def explicit_resource_type(ref: str) -> Optional[str]:
    """Return the explicit resource type encoded in ``pkg.section:name`` refs."""
    resource_ref = ref.split("#", 1)[0]
    if ":" not in resource_ref:
        return None
    prefix, _ = resource_ref.split(":", 1)
    section = prefix.rsplit(".", 1)[-1]
    return _SECTION_TO_RESOURCE_TYPE.get(section)


def resolve_resource_node(
    ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    resource_type: str,
    base_namespace: Optional[str] = None,
) -> ResourceNode:
    """
    Resolve a resource reference with package-local fallback.

    Supported forms:
    - ``pkg.section:name`` exact package ref
    - ``pkg:name`` typed shorthand
    - ``section:name`` relative to ``base_namespace`` when possible
    - ``name`` relative to ``base_namespace`` before runtime default fallback
    """
    runtime = _select_runtime(runtime=runtime, registry=registry)
    section = _RESOURCE_SECTIONS.get(resource_type, f"{resource_type}s")
    base_package = _package_from_namespace(base_namespace)

    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if ":" in ref:
        prefix, resource = ref.split(":", 1)
        if base_package and "." not in prefix and prefix in {section, resource_type}:
            add(f"{base_package}.{section}:{resource}")
        add(ref)
    else:
        if base_package:
            add(f"{base_package}.{section}:{ref}")
        add(ref)

    errors: List[str] = []
    for candidate in candidates:
        try:
            return runtime.get_node(candidate, resource_type=resource_type)
        except (KeyError, TypeError) as exc:
            errors.append(f"{candidate!r}: {exc}")

    raise KeyError(
        f"Could not resolve {resource_type} resource {ref!r}. "
        f"Tried: {candidates}. Errors: {errors}"
    )


@dataclass(frozen=True)
class AgentToolRefs:
    """Agent-config ``tools`` partitioned by the adapter they need."""

    function_refs: List[str]
    proxy_refs: List[str]


def _copy_function_tool(function: Function) -> Function:
    return function.model_copy()


def bind_function_declaration(
    declaration: Function,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
) -> Function:
    """
    Bind a prompt-local ``Function`` declaration to its package implementation.

    Prompt declarations behave like header files: they describe the schema the
    model sees. The implementation is a registered ``pkg.tools:<name>``
    resource created by ``@tool``. Matching is exact on the declaration name
    within the prompt package, then through the runtime's normal typed fallback.
    """
    if declaration.function is not None:
        return declaration

    node = resolve_resource_node(
        declaration.name,
        runtime=runtime,
        registry=registry,
        resource_type="tool",
        base_namespace=base_namespace,
    )
    implementation = node.value
    if not isinstance(implementation, Function):
        raise TypeError(
            f"Tool resource '{node.qualified_key}' is {type(implementation).__name__}, "
            "expected lllm.core.prompt.Function."
        )
    if implementation.name != declaration.name:
        raise ValueError(
            f"Tool declaration '{declaration.name}' resolved to "
            f"'{node.qualified_key}', but the implementation tool name is "
            f"'{implementation.name}'. Tool declarations bind only by exact name."
        )
    if implementation.function is None:
        raise ValueError(
            f"Tool resource '{node.qualified_key}' has no implementation. "
            "Use @tool to define the callable implementation."
        )

    return declaration.model_copy(
        update={
            "function": implementation.function,
            "processor": implementation.processor,
        }
    )


def build_registered_function(
    tool_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
) -> Function:
    """Resolve a registered ``Function`` resource and return a prompt-local copy."""
    node = resolve_resource_node(
        tool_ref,
        runtime=runtime,
        registry=registry,
        resource_type="tool",
        base_namespace=base_namespace,
    )
    function = node.value
    if not isinstance(function, Function):
        raise TypeError(
            f"Tool resource '{node.qualified_key}' is {type(function).__name__}, "
            "expected lllm.core.prompt.Function. "
            "Only @tool/Function resources are directly prompt-callable."
        )
    resource_leaf = node.key.rstrip("/").rsplit("/", 1)[-1]
    if resource_leaf != function.name:
        raise ValueError(
            f"Tool resource '{node.qualified_key}' key does not match "
            f"implementation name '{function.name}'. Tool resources bind by "
            "exact name."
        )
    if function.function is None:
        raise ValueError(
            f"Tool resource '{node.qualified_key}' has no implementation. "
            "Use @tool to define the callable implementation."
        )
    return _copy_function_tool(function)


def build_prompt_function_ref(
    tool_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
) -> Function:
    """
    Resolve a prompt-callable string ref into an executable ``Function``.

    Supported callable refs are registered ``Function`` tools and tactic
    tools. Proxy refs are intentionally rejected here; proxies need agent-level
    prompt injection through config because they expose ``query_api_doc`` and
    ``run_python`` rather than one direct function schema.
    """
    rtype = explicit_resource_type(tool_ref)
    if rtype == "proxy":
        raise TypeError(
            f"Proxy ref '{tool_ref}' cannot be placed directly in Prompt.function_list. "
            "Put it in agent config 'tools' or use the 'proxy' config block."
        )
    if rtype == "tool":
        return build_registered_function(
            tool_ref,
            runtime=runtime,
            registry=registry,
            base_namespace=base_namespace,
        )
    if rtype == "tactic":
        return build_tactic_function(
            tool_ref,
            runtime=runtime,
            registry=registry,
            base_namespace=base_namespace,
        )

    errors: Dict[str, str] = {}
    functions: Dict[str, Function] = {}

    for candidate_type, builder in (
        ("tool", build_registered_function),
        ("tactic", build_tactic_function),
    ):
        try:
            functions[candidate_type] = builder(
                tool_ref,
                runtime=runtime,
                registry=registry,
                base_namespace=base_namespace,
            )
        except Exception as exc:
            errors[candidate_type] = str(exc)

    if len(functions) == 1:
        return next(iter(functions.values()))
    if len(functions) > 1:
        raise ValueError(
            f"Tool ref '{tool_ref}' is ambiguous: it resolves as both a regular "
            "Function tool and a tactic tool. Use a full package ref such as "
            "'pkg.tools:name' or 'pkg.tactics:name'."
        )

    raise KeyError(
        f"Could not resolve prompt tool ref '{tool_ref}' as a regular Function "
        f"tool or tactic tool. Errors: {errors}"
    )


def partition_agent_tool_refs(
    refs: List[str],
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
) -> AgentToolRefs:
    """
    Split agent-config tool refs into direct prompt tools and proxy refs.

    Direct Function/tactic refs stay as strings so they are resolved lazily at
    prompt execution time. Only explicit proxy refs are resolved during agent
    build because proxy resources determine which proxy prompt/tool block must
    be injected.
    """
    runtime = _select_runtime(runtime=runtime, registry=registry)
    function_refs: List[str] = []
    proxy_refs: List[str] = []

    for ref in refs:
        if explicit_resource_type(ref) != "proxy":
            function_refs.append(ref)
            continue

        node = resolve_resource_node(
            ref,
            runtime=runtime,
            resource_type="proxy",
            base_namespace=base_namespace,
        )
        proxy_refs.append(node.qualified_key)

    return AgentToolRefs(function_refs=function_refs, proxy_refs=proxy_refs)


def _safe_basemodel_subclass(value: Any) -> Optional[Type[BaseModel]]:
    try:
        if inspect.isclass(value) and issubclass(value, BaseModel):
            return value
    except TypeError:
        return None
    return None


def _basemodel_from_annotation(annotation: Any) -> Optional[Type[BaseModel]]:
    direct = _safe_basemodel_subclass(annotation)
    if direct is not None:
        return direct

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        for arg in get_args(annotation):
            model = _safe_basemodel_subclass(arg)
            if model is not None:
                return model
    return None


def _method_type_hints(method: Callable) -> Dict[str, Any]:
    try:
        return get_type_hints(method)
    except Exception:
        return getattr(method, "__annotations__", {}) or {}


def _signature_params(method: Callable) -> list[inspect.Parameter]:
    params = []
    for param in inspect.signature(method).parameters.values():
        if param.name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        params.append(param)
    return params


def _infer_input_model(
    method: Callable, meta: TacticToolMeta
) -> Optional[Type[BaseModel]]:
    explicit = meta.input_model
    if explicit is not None:
        model = _safe_basemodel_subclass(explicit)
        if model is None:
            raise TypeError(
                "@tactictool(input_model=...) must be a pydantic BaseModel class"
            )
        return model

    hints = _method_type_hints(method)
    for param in _signature_params(method):
        model = _basemodel_from_annotation(hints.get(param.name))
        if model is not None:
            return model
    return None


def _infer_output_model(
    method: Callable, meta: TacticToolMeta
) -> Optional[Type[BaseModel]]:
    explicit = meta.output_model
    if explicit is not None:
        model = _safe_basemodel_subclass(explicit)
        if model is None:
            raise TypeError(
                "@tactictool(output_model=...) must be a pydantic BaseModel class"
            )
        return model

    return _basemodel_from_annotation(_method_type_hints(method).get("return"))


def _json_schema_for_annotation(annotation: Any) -> Dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        annotation = args[0] if args else str
        origin = get_origin(annotation)

    if origin in (list, List):
        return {"type": "array"}
    if origin in (dict, Dict):
        return {"type": "object"}

    return {"type": _PY_TYPE_TO_JSON.get(annotation, "string")}


def _schema_from_method(
    method: Callable,
    input_model: Optional[Type[BaseModel]],
    *,
    tactic_name: str,
    method_name: str,
) -> tuple[Dict[str, Any], List[str]]:
    if input_model is not None:
        schema = input_model.model_json_schema()
        return dict(schema.get("properties", {})), list(schema.get("required", []))

    hints = _method_type_hints(method)
    params = _signature_params(method)
    if not params:
        return {}, []

    properties: Dict[str, Any] = {}
    required: List[str] = []
    missing_annotations: List[str] = []
    for param in params:
        annotation = hints.get(param.name, param.annotation)
        if annotation is inspect.Parameter.empty:
            missing_annotations.append(param.name)
        prop = _json_schema_for_annotation(annotation)
        if param.default is not inspect.Parameter.empty:
            prop["description"] = f"(default: {param.default!r})"
        else:
            required.append(param.name)
        properties[param.name] = prop

    if missing_annotations:
        warnings.warn(
            f"Tactic tool '{tactic_name}.{method_name}' has unannotated "
            f"parameter(s) {missing_annotations}. Treating them as string fields.",
            RuntimeWarning,
            stacklevel=3,
        )

    return properties, required


def _decorated_tactic_tools(tactic_cls: Type) -> Dict[str, TacticToolMeta]:
    tools: Dict[str, TacticToolMeta] = {}
    for base in reversed(tactic_cls.__mro__):
        for attr_name, value in vars(base).items():
            method = (
                value.__func__
                if isinstance(value, (staticmethod, classmethod))
                else value
            )
            meta = getattr(method, "_lllm_tactic_tool", None)
            if meta is not None:
                tools[attr_name] = meta
    return tools


def _select_tactic_tool(
    tactic_cls: Type, selector: Optional[str]
) -> tuple[str, TacticToolMeta, bool]:
    tools = _decorated_tactic_tools(tactic_cls)

    if selector is not None:
        for method_name, meta in tools.items():
            names = {method_name, meta.name}
            if method_name == "call":
                names.add("root")
            if selector in names:
                return method_name, meta, False
        if not tools and selector in {"call", "root"}:
            return "call", TacticToolMeta(name="root"), True
        available = sorted(meta.name for method_name, meta in tools.items())
        raise KeyError(
            f"Tactic {tactic_cls.__name__} has no exposed tactic tool {selector!r}. "
            f"Available: {available}"
        )

    if "call" in tools:
        return "call", tools["call"], False
    if len(tools) == 1:
        method_name, meta = next(iter(tools.items()))
        return method_name, meta, False
    if len(tools) > 1:
        available = sorted(meta.name for method_name, meta in tools.items())
        raise ValueError(
            f"Tactic {tactic_cls.__name__} exposes multiple tactic tools {available}. "
            "Use a ref fragment such as 'pkg.tactics:name#tool_name'."
        )

    warnings.warn(
        f"Tactic '{getattr(tactic_cls, 'name', tactic_cls.__name__)}' has no "
        "@tactictool-decorated methods. Falling back to call(); this requires "
        "an explicit config binding from the caller.",
        RuntimeWarning,
        stacklevel=3,
    )
    return (
        "call",
        TacticToolMeta(name=getattr(tactic_cls, "name", None) or "root"),
        True,
    )


def _tool_display_name(meta: TacticToolMeta) -> str:
    return meta.name


def _tool_description(
    tactic_cls: Type, method: Callable, method_name: str, meta: TacticToolMeta
) -> str:
    description = (
        meta.description
        or (
            inspect.cleandoc(method.__doc__)
            if getattr(method, "__doc__", None)
            else None
        )
        or (inspect.cleandoc(tactic_cls.__doc__) if tactic_cls.__doc__ else None)
    )
    if description:
        return description

    generated = f"Run the '{_tool_display_name(meta)}' tactic tool."
    warnings.warn(
        f"Tactic tool '{getattr(tactic_cls, 'name', tactic_cls.__name__)}.{method_name}' "
        "has no description or docstring. "
        f"Using generated description: {generated!r}",
        RuntimeWarning,
        stacklevel=3,
    )
    return generated


def _arguments_for_method(
    spec: TacticToolSpec, kwargs: Dict[str, Any]
) -> tuple[list[Any], Dict[str, Any]]:
    if spec.input_model is not None:
        return [spec.input_model(**kwargs)], {}

    params = _signature_params(getattr(spec.tactic_cls, spec.method_name))
    if not params:
        return [], {}
    return [], kwargs


def get_tactic_tool_spec(
    tactic_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> TacticToolSpec:
    """Resolve a tactic ref and infer the metadata needed for tool exposure."""
    runtime = _select_runtime(runtime=runtime, registry=registry)
    resource_ref, selector = split_tactic_tool_ref(tactic_ref)
    node = resolve_resource_node(
        resource_ref,
        runtime=runtime,
        resource_type="tactic",
        base_namespace=base_namespace,
    )
    tactic_cls = node.value
    method_name, meta, _ = _select_tactic_tool(tactic_cls, selector)
    method = getattr(tactic_cls, method_name)

    raw_name = name or _tool_display_name(meta)
    tool_name = sanitize_tool_name(raw_name)

    tool_description = description or _tool_description(
        tactic_cls, method, method_name, meta
    )
    input_model = _infer_input_model(method, meta)
    output_model = _infer_output_model(method, meta)
    properties, required = _schema_from_method(
        method,
        input_model,
        tactic_name=getattr(tactic_cls, "name", tactic_cls.__name__),
        method_name=method_name,
    )

    return TacticToolSpec(
        ref=tactic_ref,
        tactic_ref=resource_ref,
        selector=selector,
        node=node,
        tactic_cls=tactic_cls,
        method_name=method_name,
        name=tool_name,
        description=tool_description,
        properties=properties,
        required=required,
        input_model=input_model,
        output_model=output_model,
        config=meta.config,
    )


def _config_binding(spec: TacticToolSpec, explicit_config: Any = None) -> Any:
    binding = explicit_config if explicit_config is not None else spec.config
    if binding is None:
        raise ValueError(
            f"Tactic tool '{spec.ref}' resolves to {spec.tactic_cls.__name__}, "
            f"method '{spec.method_name}', but no config binding was provided. "
            "Use @tactictool('name', config=...), pass `config=...`, "
            "or use an inline config dict."
        )
    return binding


def _materialize_config(
    spec: TacticToolSpec,
    *,
    runtime: Runtime,
    explicit_config: Any = None,
    base_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    binding = _config_binding(spec, explicit_config)
    if isinstance(binding, dict):
        return copy.deepcopy(binding)
    if isinstance(binding, str):
        config_base = (
            base_namespace if explicit_config is not None else spec.node.namespace
        )
        node = resolve_resource_node(
            binding,
            runtime=runtime,
            resource_type="config",
            base_namespace=config_base,
        )
        return resolve_config(node.qualified_key, runtime=runtime)
    raise TypeError(
        f"Tactic tool config for '{spec.ref}' must be a config URL/name or dict, "
        f"got {type(binding).__name__}"
    )


def _format_tactic_result(result: Any) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return str(result)


def call_tactic_tool(
    tactic_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
    config: Any = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve, instantiate, and call a tactic tool."""
    runtime = _select_runtime(runtime=runtime, registry=registry)
    spec = get_tactic_tool_spec(
        tactic_ref,
        runtime=runtime,
        base_namespace=base_namespace,
    )
    return _call_tactic_tool_spec(
        spec,
        runtime=runtime,
        base_namespace=base_namespace,
        config=config,
        arguments=arguments,
    )


def _call_tactic_tool_spec(
    spec: TacticToolSpec,
    *,
    runtime: Runtime,
    base_namespace: Optional[str] = None,
    config: Any = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> str:
    """Instantiate and call a tactic using an already-resolved spec."""
    tactic_config = _materialize_config(
        spec,
        runtime=runtime,
        explicit_config=config,
        base_namespace=base_namespace,
    )
    args, kwargs = _arguments_for_method(spec, dict(arguments or {}))

    from .tactic_registry import _stable_tactic_id

    tactic = spec.tactic_cls(
        tactic_config,
        registry=runtime,
        tactic_path=_stable_tactic_id(
            spec.node.namespace, getattr(spec.tactic_cls, "name", spec.node.key)
        ),
    )
    method = getattr(tactic, spec.method_name)
    return _format_tactic_result(method(*args, **kwargs))


def build_tactic_function(
    tactic_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
    config: Any = None,
) -> Function:
    """Build an executable ``Function`` that calls the referenced tactic."""
    runtime = _select_runtime(runtime=runtime, registry=registry)
    spec = get_tactic_tool_spec(
        tactic_ref,
        runtime=runtime,
        base_namespace=base_namespace,
    )
    _config_binding(spec, config)

    def _call_tactic(**kwargs: Any) -> str:
        return _call_tactic_tool_spec(
            spec,
            runtime=runtime,
            base_namespace=base_namespace,
            config=config,
            arguments=kwargs,
        )

    return Function(
        name=spec.name,
        description=spec.description,
        properties=copy.deepcopy(spec.properties),
        required=list(spec.required),
        function=_call_tactic,
    )


def _example_for_property(prop: Dict[str, Any]) -> Any:
    if "default" in prop:
        return prop["default"]
    if "examples" in prop and prop["examples"]:
        return prop["examples"][0]
    json_type = prop.get("type", "string")
    if json_type == "integer":
        return 1
    if json_type == "number":
        return 1.0
    if json_type == "boolean":
        return True
    if json_type == "array":
        return []
    if json_type == "object":
        return {}
    return prop.get("description", "")


def build_tactic_endpoint_info(
    tactic_ref: str,
    *,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
    endpoint: Optional[str] = None,
    category: str = "tactics",
    name: Optional[str] = None,
    description: Optional[str] = None,
    response: Any = None,
    method: str = "TACTIC",
) -> Dict[str, Any]:
    """Build ``BaseProxy.endpoint``-compatible metadata for a tactic."""
    spec = get_tactic_tool_spec(
        tactic_ref,
        runtime=runtime,
        registry=registry,
        base_namespace=base_namespace,
        name=name,
        description=description,
    )

    params: Dict[str, Any] = {}
    for prop_name, prop in spec.properties.items():
        param_name = f"{prop_name}*" if prop_name in spec.required else prop_name
        py_type = _JSON_TYPE_TO_PYTHON.get(prop.get("type", "string"), str)
        params[param_name] = (py_type, _example_for_property(prop))

    if response is None:
        response = (
            spec.output_model.model_json_schema()
            if spec.output_model is not None
            else "str"
        )

    return {
        "category": category,
        "endpoint": endpoint or spec.name,
        "name": name or spec.name,
        "description": description or spec.description,
        "sub_category": None,
        "remove_keys": None,
        "params": params,
        "response": response,
        "dt_cutoff": None,
        "method": method,
    }


def call_tactic_proxy_endpoint(
    tactic_ref: str,
    *,
    params: Any = None,
    kwargs: Optional[Dict[str, Any]] = None,
    runtime: Optional[Runtime] = None,
    registry: Optional[Registry] = None,
    base_namespace: Optional[str] = None,
    config: Any = None,
) -> str:
    """Call a tactic endpoint from a proxy method."""
    runtime = _select_runtime(runtime=runtime, registry=registry)
    kwargs = dict(kwargs or {})

    if params is None:
        arguments = kwargs
    elif isinstance(params, dict):
        arguments = dict(params)
        arguments.update(kwargs)
    else:
        arguments = {"task": params}
        arguments.update(kwargs)

    return call_tactic_tool(
        tactic_ref,
        runtime=runtime,
        registry=registry,
        base_namespace=base_namespace,
        config=config,
        arguments=arguments,
    )
