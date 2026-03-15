import random
import time
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field

from lllm.core.prompt import Prompt, AgentException, AgentCallSession
from lllm.core.const import Roles, APITypes
from lllm.core.dialog import Dialog, Message
from lllm.core.log import ReplayableLogBase
from lllm.invokers.base import BaseInvoker, BaseStreamHandler
import lllm.utils as U



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

