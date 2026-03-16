# lllm/__init__.py
from lllm.core.runtime import (
    Runtime, get_default_runtime, set_default_runtime,
    get_runtime, load_runtime,
)
from lllm.core.resource import (
    ResourceNode, PackageInfo,
    load_prompt, load_tactic, load_proxy, load_config, load_resource,
)
from lllm.core.config import (
    load_package, find_config_file,
    resolve_config, AgentSpec, parse_agent_configs,
)
from lllm.core.prompt import Function, FunctionCall, MCP, Prompt
from lllm.core.dialog import Message, Dialog
from lllm.core.agent import Agent
from lllm.core.tactic import Tactic, build_tactic, register_tactic_class
from lllm.core.const import Roles, Modalities, Invokers, APITypes
from lllm.proxies import BaseProxy, Proxy, register_proxy, ProxyRegistrator

__version__ = "0.0.2"


def _auto_init():
    rt = get_default_runtime()
    if rt._discovery_done:
        return
    load_runtime()

_auto_init()