from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import inspect
import datetime as dt
from enum import Enum
import logging
logging.basicConfig(level=logging.INFO)

from lllm.core.agent import Agent
from lllm.core.runtime import Runtime, get_default_runtime
from lllm.core.config import auto_discover_if_enabled
from lllm.invokers import build_invoker
import lllm.utils as U
from lllm.core.const import APITypes
from lllm.core.log import build_log_base



# ---------------------------------------------------------------------------
# Agent registration and building
# ---------------------------------------------------------------------------



def _normalize_agent_type(agent_type):
    if isinstance(agent_type, Enum) or (isinstance(agent_type, type) and issubclass(agent_type, Enum)):
        return agent_type.value
    elif isinstance(agent_type, str):
        return agent_type
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

def register_agent_class(agent_cls: Type['Orchestra'], runtime: Runtime = None) -> Type['Orchestra']:
    runtime = runtime or get_default_runtime()
    agent_type = _normalize_agent_type(getattr(agent_cls, 'agent_type', None))
    assert agent_type not in (None, ''), f"Agent class {agent_cls.__name__} must define `agent_type`"
    if agent_type in runtime.agents and runtime.agents[agent_type] is not agent_cls:
        raise ValueError(f"Agent type '{agent_type}' already registered with {runtime.agents[agent_type].__name__}")
    runtime.register_agent(agent_type, agent_cls)
    return agent_cls

def get_agent_class(agent_type: str, runtime: Runtime = None) -> Type['Orchestra']:
    runtime = runtime or get_default_runtime()
    if agent_type not in runtime.agents:
        raise KeyError(f"Agent type '{agent_type}' not found. Registered: {list(runtime.agents.keys())}")
    return runtime.agents[agent_type]

def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: str = None, runtime: Runtime = None, **kwargs) -> 'Orchestra':
    if agent_type is None:
        agent_type = config.get('agent_type')
    agent_type = _normalize_agent_type(agent_type)
    agent_cls = get_agent_class(agent_type, runtime)
    return agent_cls(config, ckpt_dir, stream, **kwargs)

class Orchestra:
    """
    Orchestra is the **Core** base class for LLLM.
    It is used to create custom agents. It is responsible for:
    - Initializing the agents by reading the agent configs, you should designate which configs to read by setting the `agent_group` attribute.
    """
    agent_type: str | Enum = None
    agent_group: List[str] = None
    is_async: bool = False

    def __init_subclass__(cls, register: bool = True, runtime: Optional[Runtime] = None, **kwargs):
        runtime = runtime or get_default_runtime()
        super().__init_subclass__(**kwargs)
        if register:
            register_agent_class(cls, runtime=runtime)

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream = None, runtime: Optional[Runtime] = None):
        self._runtime = runtime or get_default_runtime()
        auto_discover_if_enabled(config.get("auto_discover"), runtime=self._runtime)
        if stream is None:
            stream = U.PrintSystem()
        self.config = config
        assert self.agent_group is not None, f"Agent group is not set for {self.agent_type}"
        _agent_configs = config['agent_configs']
        self.agent_configs = {}
        for agent_name in self.agent_group:
            assert agent_name in _agent_configs, f"Agent {agent_name} not found in agent configs"
            self.agent_configs[agent_name] = _agent_configs[agent_name]
        self._stream = stream
        self._stream_backup = stream
        self.st = None
        self.ckpt_dir = ckpt_dir
        self._log_base = build_log_base(config)
        self.agents = {}

        # Initialize Invoker via runtime
        self.llm_invoker = build_invoker(config)

        for agent_name, model_config in self.agent_configs.items():
            model_config = model_config.copy()
            self.model = model_config.pop('model_name')
            system_prompt_path = model_config.pop('system_prompt_path')
            api_type_value = model_config.pop('api_type', APITypes.COMPLETION.value)
            if isinstance(api_type_value, APITypes):
                api_type = api_type_value
            else:
                api_type = APITypes(api_type_value)
            
            self.agents[agent_name] = Agent(
                name=agent_name,
                system_prompt=self._runtime.get_prompt(system_prompt_path),
                model=self.model,
                llm_invoker=self.llm_invoker,
                api_type=api_type,
                model_args=model_config,
                log_base=self._log_base,
                max_exception_retry=self.config.get('max_exception_retry', 3),
                max_interrupt_steps=self.config.get('max_interrupt_steps', 5),
                max_llm_recall=self.config.get('max_llm_recall', 0),
            )

        self.__additional_args = {}
        sig = inspect.signature(self.call)
        for arg in sig.parameters:
            if arg not in {'task', '**kwargs'}:
                self.__additional_args[arg] = sig.parameters[arg].default

    def set_st(self, session_name: str):
        self.st = U.StreamWrapper(self._stream, self._log_base, session_name)

    def restore_st(self):
        pass

    def silent(self):
        self._stream = U.PrintSystem(silent=True)

    def restore(self):
        self._stream = self._stream_backup

    def call(self, task: str, **kwargs):
        raise NotImplementedError

    def __call__(self, task: str, session_name: str = None, **kwargs) -> str:
        if session_name is None:
            session_name = task.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.set_st(session_name)
        report = self.call(task, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report
