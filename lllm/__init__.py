# lllm/__init__.py
import os

from .core.agent import Agent
from .core.config import (
    AgentSpec,
    find_config_file,
    load_cwd_fallback,
    load_package,
    parse_agent_configs,
    resolve_config,
)
from .core.dialog import Dialog, Message
from .core.prompt import MCP, Function, FunctionCall, Prompt, tool
from .core.resource import (
    PackageInfo,
    ResourceNode,
    load_config,
    load_prompt,
    load_proxy,
    load_resource,
    load_tactic,
    load_tool,
)
from .core.runtime import (
    Runtime,
    export_package,
    get_default_runtime,
    get_runtime,
    install_package,
    list_packages,
    load_runtime,
    remove_package,
    set_default_runtime,
)
from .core.tactic import Tactic, build_tactic, register_tactic_class
from .core.tactic_tool import tactictool
from .logging import (
    LocalFileBackend,
    LogStore,
    NoOpBackend,
    SQLiteBackend,
    setup_logging,
)
from .proxies import BaseProxy, ProxyManager, ProxyRegistrator, register_proxy

__version__ = "0.1.1"


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _auto_init():
    if not _env_flag_enabled("LLLM_AUTO_INIT", default=True):
        return
    rt = get_default_runtime()
    if rt.discovery_done:
        return
    load_runtime(
        discover_cwd=_env_flag_enabled("LLLM_AUTO_CWD_FALLBACK", default=True),
        discover_shared_packages=_env_flag_enabled(
            "LLLM_AUTO_SHARED_PACKAGES",
            default=True,
        ),
    )


_auto_init()
