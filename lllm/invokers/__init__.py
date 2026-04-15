from __future__ import annotations

from typing import Callable, Dict

from lllm.invokers.base import BaseInvoker

InvokerBuilder = Callable[[Dict], BaseInvoker]


def _build_litellm_invoker(config: Dict) -> BaseInvoker:
    from lllm.invokers.litellm import LiteLLMInvoker

    return LiteLLMInvoker(config)


_PROVIDER_BUILDERS: Dict[str, InvokerBuilder] = {
    "litellm": _build_litellm_invoker,
}


def register_invoker(name: str, builder: InvokerBuilder, *, overwrite: bool = False) -> None:
    name = name.lower()
    if name in _PROVIDER_BUILDERS and not overwrite:
        raise ValueError(f"Invoker '{name}' already registered")
    _PROVIDER_BUILDERS[name] = builder


def build_invoker(config: Dict) -> BaseInvoker:
    invoker_name = config.get("invoker", "litellm").lower()
    try:
        builder = _PROVIDER_BUILDERS[invoker_name]
    except KeyError as exc:
        raise KeyError(
            f"Invoker '{invoker_name}' not registered. Available: {sorted(_PROVIDER_BUILDERS)}"
        ) from exc
    invoker_config = config.get("invoker_config", config)
    return builder(invoker_config)


__all__ = ["register_invoker", "build_invoker", "InvokerBuilder"]
