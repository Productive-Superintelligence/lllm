import random
import time
import json
import uuid
import inspect
import datetime as dt
import numpy as np
from typing import List, Dict, Any, Tuple, Type, Optional
from dataclasses import dataclass, field
from enum import Enum

from lllm.core.models import Message, Prompt, FunctionCall, AgentException, Function
from lllm.core.const import Roles, APITypes
from lllm.core.dialog import Dialog
from lllm.core.log import ReplayableLogBase, build_log_base
from lllm.invokers.base import BaseInvoker, BaseStreamHandler
import lllm.utils as U
from lllm.core.discovery import auto_discover_if_enabled
from lllm.invokers import build_invoker
from lllm.core.context import Context, get_default_context


def _normalize_agent_type(agent_type):
    if isinstance(agent_type, Enum) or (isinstance(agent_type, type) and issubclass(agent_type, Enum)):
        return agent_type.value
    elif isinstance(agent_type, str):
        return agent_type
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

def register_agent_class(agent_cls: Type['Orchestrator'], context: Context = None) -> Type['Orchestrator']:
    ctx = context or get_default_context()
    agent_type = _normalize_agent_type(getattr(agent_cls, 'agent_type', None))
    assert agent_type not in (None, ''), f"Agent class {agent_cls.__name__} must define `agent_type`"
    if agent_type in ctx.agents and ctx.agents[agent_type] is not agent_cls:
        raise ValueError(f"Agent type '{agent_type}' already registered with {ctx.agents[agent_type].__name__}")
    ctx.register_agent(agent_type, agent_cls)
    return agent_cls

def get_agent_class(agent_type: str, context: Context = None) -> Type['Orchestrator']:
    ctx = context or get_default_context()
    if agent_type not in ctx.agents:
        raise KeyError(f"Agent type '{agent_type}' not found. Registered: {list(ctx.agents.keys())}")
    return ctx.agents[agent_type]

def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: str = None, context: Context = None, **kwargs) -> 'Orchestrator':
    if agent_type is None:
        agent_type = config.get('agent_type')
    agent_type = _normalize_agent_type(agent_type)
    agent_cls = get_agent_class(agent_type, context)
    return agent_cls(config, ckpt_dir, stream, **kwargs)

@dataclass
class Agent:
    name: str # the role of the agent, or a name of the agent
    system_prompt: Prompt
    model: str # the model identifier (e.g., 'gpt-4o'), by default, it from litellm model list (https://models.litellm.ai/)
    llm_invoker: BaseInvoker
    stream_handler: Optional[BaseStreamHandler] = None
    log_base: Optional[ReplayableLogBase] = None
    api_type: APITypes = APITypes.COMPLETION
    model_args: Dict[str, Any] = field(default_factory=dict) # additional args, like temperature, seed, etc.
    max_exception_retry: int = 3
    max_interrupt_times: int = 5
    max_llm_recall: int = 0

    """
    Represents a single LLM agent with a specific role and capabilities.

    Attributes:
        name (str): The name or role of the agent (e.g., 'assistant', 'coder').
        system_prompt (Prompt): The system prompt defining the agent's persona.
        model (str): The model identifier (e.g., 'gpt-4o').
        llm_invoker (BaseInvoker): The invoker instance for LLM calls.
        log_base (ReplayableLogBase): Logger for recording interactions.
        model_args (Dict[str, Any]): Additional model arguments (temp, top_p, etc.).
        max_exception_retry (int): Max retries for agent exceptions.
        max_interrupt_times (int): Max consecutive tool call interrupts.
        max_llm_recall (int): Max retries for LLM API errors.
    """

    def reload_system(self, system_prompt: Prompt):
        self.system_prompt = system_prompt
        
    # initialize the dialog
    def init_dialog(self, prompt_args: Optional[Dict[str, Any]] = None, session_name: str = None) -> Dialog:
        prompt_args = dict(prompt_args) if prompt_args else {}
        if session_name is None:
            session_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(uuid.uuid4())[:6]
        system_message = Message(
            role=Roles.SYSTEM,
            content=self.system_prompt(**prompt_args),
            name='system',
        )
        return Dialog(
            _messages=[system_message],
            session_name=session_name,
            log_base=self.log_base,
            top_prompt=self.system_prompt,
        )

    # send a message to the dialog manually
    def send_message(
        self,
        dialog: Dialog,
        prompt: Prompt,
        prompt_args: Optional[Dict[str, Any]] = None,
        sender_name: str = 'internal',
        extra: Optional[Dict[str, Any]] = None,
        role: Roles = Roles.USER,
    ):
        prompt_payload = dict(prompt_args) if prompt_args else None
        extra_payload = dict(extra) if extra else None
        return dialog.send_message(prompt, prompt_payload, name=sender_name, extra=extra_payload, role=role)

    # it performs the "Agent Call"
    def call(
        self,
        dialog: Dialog,  # it assumes the prompt is already loaded into the dialog as the top prompt by send_message
        extra: Optional[Dict[str, Any]] = None,  # for tracking additional information, such as frontend replay info
        args: Optional[Dict[str, Any]] = None,  # for tracking additional information, such as frontend replay info
        parser_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Message, Dialog, List[FunctionCall]]:
        """
        Executes the agent loop, handling LLM calls, tool execution, and interrupts.

        Args:
            dialog (Dialog): The current dialog state.
            extra (Dict[str, Any], optional): Extra metadata for the call.
            args (Dict[str, Any], optional): Additional arguments for the prompt.
            parser_args (Dict[str, Any], optional): Arguments for the output parser.

        Returns:
            Tuple[Message, Dialog, List[FunctionCall]]: The final response message, the updated dialog, and a list of executed function calls.

        Raises:
            ValueError: If the agent fails to produce a valid response after retries.
        """
        extra = dict(extra) if extra else {}
        args = dict(args) if args else {}
        parser_args = dict(parser_args) if parser_args else {}
        # Prompt: a function maps prompt args and dialog into the expected output 
        if dialog.top_prompt is None:
            dialog.top_prompt = self.system_prompt
        interrupts = []
        for i in range(10000 if self.max_interrupt_times == 0 else self.max_interrupt_times+1): # +1 for the final response
            llm_recall = self.max_llm_recall 
            exception_retry = self.max_exception_retry 
            working_dialog = dialog.fork() # make a copy of the dialog, truncate all excception handling dialogs
            while True: # ensure the response is no exception
                execution_attempts = []
                try:
                    _model_args = self.model_args.copy()
                    _model_args.update(args)
                    
                    response = self.llm_invoker.call(
                        working_dialog,
                        self.model,
                        _model_args,
                        parser_args=parser_args,
                        responder=self.name,
                        extra=extra,
                        api_type=self.api_type,
                        stream_handler=self.stream_handler,
                    )
                    working_dialog.append(response) 
                    if response.execution_errors != []:
                        execution_attempts.append(response)
                        raise AgentException(response.error_message)
                    else: 
                        break
                except AgentException as e: # handle the exception from the agent
                    if exception_retry > 0:
                        exception_retry -= 1
                        U.cprint(f'{self.name} is handling an exception {e}, retry times: {self.max_exception_retry-exception_retry}/{self.max_exception_retry}','r')
                        working_dialog.send_message(dialog.top_prompt.exception_handler, {'error_message': str(e)}, name='exception')
                        continue
                    else:
                        raise e
                except Exception as e: # handle the exception from the LLM
                    # Simplified error handling for now
                    wait_time = random.random()*15+1
                    if U.is_openai_rate_limit_error(e): # for safe
                        time.sleep(wait_time)
                    else:
                        if llm_recall > 0:
                            llm_recall -= 1
                            time.sleep(1) # wait for a while before retrying
                            continue
                        else:
                            raise e

            response.execution_attempts = execution_attempts
            dialog.append(response) # update the dialog state
            # now handle the interruption
            if response.is_function_call:
                _func_names = [func_call.name for func_call in response.function_calls]
                U.cprint(f'{self.name} is calling function {_func_names}, interrupt times: {i+1}/{self.max_interrupt_times}','y')
                # handle the function call
                for function_call in response.function_calls:
                    if function_call.is_repeated(interrupts):
                        result_str = f'The function {function_call.name} with identical arguments {function_call.arguments} has been called earlier, please check the previous results and do not call it again. If you do not need to call more functions, just stop calling and provide the final response.'
                    else:
                        print(f'{self.name} is calling function {function_call.name} with arguments {function_call.arguments}')
                        if function_call.name not in dialog.top_prompt.functions:
                            raise KeyError(f"Function '{function_call.name}' not registered on prompt '{dialog.top_prompt.path}'")
                        function = dialog.top_prompt.functions[function_call.name]
                        function_call = function(function_call)
                        result_str = function_call.result_str
                        interrupts.append(function_call)
                    dialog.send_message(
                        dialog.top_prompt.interrupt_handler,
                        {'call_results': result_str},
                        role=Roles.TOOL,
                        name=function_call.name,
                        extra={'tool_call_id': function_call.id},
                    )
                if i == self.max_interrupt_times-1:
                    dialog.send_message(dialog.top_prompt.interrupt_handler_final, role=Roles.USER, name=function_call.name)
            else: # the response is not a function call, it is the final response
                if i > 0:   
                    U.cprint(f'{self.name} stopped calling functions, total interrupt times: {i}/{self.max_interrupt_times}','y')
                return response, dialog, interrupts
        raise ValueError('Failed to call the agent')




class Orchestrator:
    """
    Orchestrator is the **Core** base class for LLLM.
    It is used to create custom agents. It is responsible for:
    - Initializing the agents by reading the agent configs, you should designate which configs to read by setting the `agent_group` attribute.
    """
    agent_type: str | Enum = None
    agent_group: List[str] = None
    is_async: bool = False

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            register_agent_class(cls)

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream = None, context: Optional[Context] = None):
        self._context = context or get_default_context()
        auto_discover_if_enabled(config.get("auto_discover"), context=self._context)
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

        # Initialize Invoker via context
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
                system_prompt=self._context.get_prompt(system_prompt_path),
                model=self.model,
                llm_invoker=self.llm_invoker,
                api_type=api_type,
                model_args=model_config,
                log_base=self._log_base,
                max_exception_retry=self.config.get('max_exception_retry', 3),
                max_interrupt_times=self.config.get('max_interrupt_times', 5),
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
