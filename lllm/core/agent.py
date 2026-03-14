import random
import time
import json
import uuid
import inspect
import datetime as dt
import numpy as np
from typing import List, Dict, Any, Tuple, Type, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from lllm.core.prompt import Prompt, AgentException, AgentCallSession
from lllm.core.const import Roles, APITypes
from lllm.core.dialog import Dialog, Message
from lllm.core.log import ReplayableLogBase, build_log_base
from lllm.invokers.base import BaseInvoker, BaseStreamHandler, InvokeResult
import lllm.utils as U
from lllm.core.discovery import auto_discover_if_enabled
from lllm.invokers import build_invoker
from lllm.core.runtime import Runtime, get_default_runtime


def _normalize_agent_type(agent_type):
    if isinstance(agent_type, Enum) or (isinstance(agent_type, type) and issubclass(agent_type, Enum)):
        return agent_type.value
    elif isinstance(agent_type, str):
        return agent_type
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

def register_agent_class(agent_cls: Type['Orchestrator'], runtime: Runtime = None) -> Type['Orchestrator']:
    runtime = runtime or get_default_runtime()
    agent_type = _normalize_agent_type(getattr(agent_cls, 'agent_type', None))
    assert agent_type not in (None, ''), f"Agent class {agent_cls.__name__} must define `agent_type`"
    if agent_type in runtime.agents and runtime.agents[agent_type] is not agent_cls:
        raise ValueError(f"Agent type '{agent_type}' already registered with {runtime.agents[agent_type].__name__}")
    runtime.register_agent(agent_type, agent_cls)
    return agent_cls

def get_agent_class(agent_type: str, runtime: Runtime = None) -> Type['Orchestrator']:
    runtime = runtime or get_default_runtime()
    if agent_type not in runtime.agents:
        raise KeyError(f"Agent type '{agent_type}' not found. Registered: {list(runtime.agents.keys())}")
    return runtime.agents[agent_type]

def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: str = None, runtime: Runtime = None, **kwargs) -> 'Orchestrator':
    if agent_type is None:
        agent_type = config.get('agent_type')
    agent_type = _normalize_agent_type(agent_type)
    agent_cls = get_agent_class(agent_type, runtime)
    return agent_cls(config, ckpt_dir, stream, **kwargs)



@dataclass
class Agent:
    """
    Represents a single LLM agent with a specific role and capabilities.

    An Agent owns the dialogs it creates. Each dialog is keyed by a
    user-chosen alias (e.g. 'planning', 'talk_with_coder') that makes the
    code self-documenting:

        agent.open('planning', prompt_args={...})
        agent.receive("What's the plan?")
        response = agent.respond()

        agent.open('execution', prompt_args={...})
        agent.switch('execution')
        ...

    For power-user / cross-agent scenarios, ``call(dialog)`` still accepts
    a raw Dialog directly — but the recommended path is alias-based.

    Attributes:
        name: The name or role of the agent (e.g., 'assistant', 'coder').
        system_prompt: The system prompt defining the agent's persona.
        model: The model identifier (e.g., 'gpt-4o').
        llm_invoker: The invoker instance for LLM calls.
        model_args: Additional model arguments (temp, top_p, etc.).
        max_exception_retry: Max retries for agent parsing/validation exceptions.
        max_interrupt_steps: Max consecutive tool call interrupts.
        max_llm_recall: Max retries for LLM API errors.
    """
    name: str # the role of the agent, or a name of the agent
    system_prompt: Prompt
    model: str # the model identifier (e.g., 'gpt-4o'), by default, it from litellm model list (https://models.litellm.ai/)
    llm_invoker: BaseInvoker
    stream_handler: Optional[BaseStreamHandler] = None
    log_base: Optional[ReplayableLogBase] = None
    api_type: APITypes = APITypes.COMPLETION
    model_args: Dict[str, Any] = field(default_factory=dict) # additional args, like temperature, seed, etc.
    max_exception_retry: int = 3
    max_interrupt_steps: int = 5
    max_llm_recall: int = 0

    # Dialog management
    _dialogs: Dict[str, Dialog] = field(default_factory=dict, repr=False)
    _active_alias: Optional[str] = field(default=None, repr=False)

    def open(self, alias: str, prompt_args=None, session_name=None):
        """Create a new dialog owned by this agent, keyed by alias."""
        if alias in self._dialogs:
            raise ValueError(
                f"Dialog '{alias}' already exists on agent '{self.name}'. "
                f"Use .fork('{alias}', ...) or .close('{alias}') first."
            )
        prompt_args = dict(prompt_args) if prompt_args else {}
        dialog = Dialog(
            session_name=session_name or f"{self.name}_{alias}",
            log_base=self.log_base,
            owner=self.name,
        )
        dialog.put_prompt(
            self.system_prompt, prompt_args,
            name='system', role=Roles.SYSTEM,
        )
        self._dialogs[alias] = dialog
        self._active_alias = alias
        return self # for chaining
        
    def fork(self, alias: str, child_alias: str, last_n: int = 0, first_k: int = 1, switch: bool = False) -> 'Agent':
        """
        Branch an existing dialog into a new child dialog.

        The parent dialog's ``fork()`` handles all lineage bookkeeping
        (parent ↔ child links, split_point, ids).  Agent just stores
        the child under ``child_alias`` and switches to it.

        Args:
            alias: the source dialog to fork from.
            child_alias: the alias for the new child dialog.
            last_n: if >0, drop the last n messages from the copy.
            first_k: if >0, keep the first k messages from the copy. Only used when last_n is >0.
            switch: if True, switch to the new child dialog after forking.

        Raises:
            ValueError: if ``child_alias`` is already in use.
            KeyError: if ``alias`` doesn't exist.
        """
        if child_alias in self._dialogs:
            raise ValueError(
                f"Dialog '{child_alias}' already exists on agent '{self.name}'."
            )
        parent = self._get_dialog(alias)
        child = parent.fork(last_n, first_k)
        self._dialogs[child_alias] = child
        if switch:
            self._active_alias = child_alias
        return self # for chaining

    def close(self, alias: str) -> Dialog:
        """
        Remove a dialog from this agent and return it.

        Useful for archiving, handing off to another system, or just
        cleaning up.  If the closed dialog was active, active becomes None.
        """
        dialog = self._dialogs.pop(alias)
        if self._active_alias == alias:
            self._active_alias = None
        return dialog

    def switch(self, alias: str) -> 'Agent':
        """
        Set the active dialog by alias.  Returns self for chaining.

        Raises:
            KeyError: if ``alias`` doesn't exist.
        """
        if alias not in self._dialogs:
            raise KeyError(
                f"No dialog '{alias}' on agent '{self.name}'. "
                f"Available: {list(self._dialogs.keys())}"
            )
        self._active_alias = alias
        return self

    def _get_dialog(self, alias: str = None) -> Dialog:
        """Resolve alias → Dialog, falling back to active dialog if alias is None."""
        if alias is not None:
            if alias not in self._dialogs:
                raise KeyError(
                    f"No dialog '{alias}' on agent '{self.name}'. "
                    f"Available: {list(self._dialogs.keys())}"
                )
            return self._dialogs[alias]
        if self._active_alias is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no active dialog. "
                f"Call .open(alias) or .switch(alias) first."
            )
        return self._dialogs[self._active_alias]

    @property
    def current_dialog(self) -> Dialog:
        """The currently active dialog."""
        return self._get_dialog()

    @property
    def dialogs(self) -> Dict[str, Dialog]:
        """Read-only snapshot of all managed dialogs (alias → Dialog)."""
        return dict(self._dialogs)

    @property
    def active_alias(self) -> Optional[str]:
        return self._active_alias

    # ===================================================================
    # Messaging primitives — operate on active or specified dialog
    # ===================================================================

    def receive(
        self,
        text: str,
        alias: str = None,
        role: Roles = Roles.USER,
        name: str = 'user',
    ) -> Message:
        """Put a text message into the active (or specified) dialog."""
        return self._get_dialog(alias).put_text(text, name=name, role=role)

    def receive_prompt(
        self,
        prompt: Prompt,
        prompt_args: Optional[Dict[str, Any]] = None,
        alias: str = None,
        role: Roles = Roles.USER,
        name: str = 'user',
    ) -> Message:
        """Put a structured prompt message into the dialog."""
        return self._get_dialog(alias).put_prompt(
            prompt, prompt_args, name=name, role=role,
        )

    def receive_image(
        self,
        image,
        caption: str = None,
        alias: str = None,
        role: Roles = Roles.USER,
        name: str = 'user',
    ) -> Message:
        """Put an image message into the dialog."""
        return self._get_dialog(alias).put_image(
            image, caption=caption, name=name, role=role,
        )

    def respond(
        self,
        alias: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        return_session: bool = False,
    ) -> Union[Message, Tuple[Message, AgentCallSession]]:
        """
        High-level: run the agent call loop on a dialog, return the response.

        This is the recommended way to get a response.  For full diagnostics
        (call_state with retry info, model_args, etc.), use ``call()`` directly.

        Args:
            alias: the alias of the dialog to respond to.
            metadata: additional metadata for the call.
            args: additional arguments for the prompt.
            parser_args: arguments for the output parser.
            return_session: if True, return the entire call session instead of just the message, 
            which includes the retry info, model args, etc. You can use session.delivery to get the final message.
        """
        dialog = self._get_dialog(alias)
        session = self._call(dialog, metadata=metadata, args=args, parser_args=parser_args)
        if return_session:
            return session
        else:
            return session.delivery


    # ===================================================================
    # Core agent call loop
    # ===================================================================

    # it performs the "Agent Call"
    def _call(
        self,
        dialog: Dialog,  # it assumes the prompt is already loaded into the dialog as the top prompt by send_message
        metadata: Optional[Dict[str, Any]] = None,  # for tracking additional information, such as frontend replay info
        args: Optional[Dict[str, Any]] = None,  # for tracking additional information, such as frontend replay info
        parser_args: Optional[Dict[str, Any]] = None,
    ) -> AgentCallSession:
        """
        Executes the agent loop, handling LLM calls, tool execution, and interrupts.

        Args:
            dialog (Dialog): The current dialog state.
            metadata (Dict[str, Any], optional): Extra metadata for the call.
            args (Dict[str, Any], optional): Additional arguments for the prompt.
            parser_args (Dict[str, Any], optional): Arguments for the output parser.

        Returns:
            Tuple[Message, Dialog, List[FunctionCall]]: The final response message, the updated dialog, and a list of executed function calls.

        Raises:
            ValueError: If the agent fails to produce a valid response after retries.
        """
        session = AgentCallSession(
            agent_name=self.name,
            max_exception_retry=self.max_exception_retry,
            max_interrupt_steps=self.max_interrupt_steps,
            max_llm_recall=self.max_llm_recall,
        )
        metadata = dict(metadata) if metadata else {}
        args = dict(args) if args else {}
        parser_args = dict(parser_args) if parser_args else {}
        # Prompt: a function maps prompt args and dialog into the expected output 
        if dialog.top_prompt is None:
            dialog.top_prompt = self.system_prompt
        interrupts = []
        for i in range(10000 if self.max_interrupt_steps == 0 else self.max_interrupt_steps+1): # +1 for the final response
            working_dialog = dialog.fork() # make a copy of the dialog, truncate all excception handling dialogs
            while True: # ensure the response is no exception
                try:
                    _model_args = self.model_args.copy()
                    _model_args.update(args)
                    
                    invoke_result = self.llm_invoker.call(
                        working_dialog,
                        self.model,
                        _model_args,
                        parser_args=parser_args,
                        responder=self.name,
                        metadata=metadata,
                        api_type=self.api_type,
                        stream_handler=self.stream_handler,
                    )
                    session.new_invoke_trace(invoke_result, i)
                    working_dialog.append(invoke_result.message) 
                    if invoke_result.has_errors:
                        raise AgentException(invoke_result.error_message)
                    else: 
                        break
                except AgentException as e: # handle the exception from the agent
                    if not session.reach_max_exception_retry:
                        session.exception(e, i)
                        working_dialog.put_prompt(
                            dialog.top_prompt.on_exception(session), 
                            {'error_message': str(e)}, 
                            name='exception'
                        )
                        continue
                    else:
                        raise e
                except Exception as e: # handle the exception from the LLM
                    # Simplified error handling for now
                    wait_time = random.random()*15+1
                    if U.is_openai_rate_limit_error(e): # for safe
                        time.sleep(wait_time)
                    else:
                        if not session.reach_max_llm_recall:
                            session.llm_recall(e, i)
                            time.sleep(1) # wait for a while before retrying
                            continue
                        else:
                            raise e

            dialog.append(invoke_result.message) # update the dialog state
            # now handle the interruption
            if invoke_result.message.is_function_call:
                _func_names = [func_call.name for func_call in invoke_result.message.function_calls]
                # handle the function call
                session.interrupt(invoke_result.message.function_calls, i)
                for function_call in invoke_result.message.function_calls:
                    if function_call.is_repeated(interrupts):
                        result_str = f'The function {function_call.name} with identical arguments {function_call.arguments} has been called earlier, please check the previous results and do not call it again. If you do not need to call more functions, just stop calling and provide the final response.'
                    else:
                        if function_call.name not in dialog.top_prompt.functions:
                            raise KeyError(f"Function '{function_call.name}' not registered on prompt '{dialog.top_prompt.path}'")
                        function = dialog.top_prompt.functions[function_call.name]
                        function_call = function(function_call)
                        result_str = function_call.result_str
                        interrupts.append(function_call)
                    dialog.put_prompt(
                        dialog.top_prompt.on_interrupt(session),
                        {'call_results': result_str},
                        role=Roles.TOOL,
                        name=function_call.name,
                        metadata={'tool_call_id': function_call.id},
                    )
                
                if session.reach_max_interrupt_steps:
                    dialog.put_prompt(
                        dialog.top_prompt.on_interrupt_final(session), 
                        role=Roles.USER, 
                        name=function_call.name
                    )
            else: # the response is not a function call, it is the final response
                session.success(invoke_result.message)
                return session
        session.failure(self.log_base)
        raise ValueError(f'Failed to call the agent: {session}')




class Orchestrator:
    """
    Orchestrator is the **Core** base class for LLLM.
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
