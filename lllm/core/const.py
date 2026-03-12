from enum import Enum
from pydantic import BaseModel
import logging
logging.basicConfig(level=logging.INFO)

class RCollections(str, Enum):
    DIALOGS = 'dialogs'
    FRONTEND = 'frontend'
    MESSAGES = 'messages'

class ParseError(Exception):
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.message)

class Roles(str, Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    TOOL = 'tool' # only use tool role, not function role
    TOOL_CALL = 'tool_call'

    @property
    def msg_value(self):
        if self == Roles.SYSTEM:
            return 'developer'
        return self.value

class Invokers(str, Enum):
    LITELLM = 'litellm'

class Modalities(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    FUNCTION_CALL = 'function_call'

class APITypes(str, Enum):
    COMPLETION = 'completion'
    RESPONSE = 'response'

LLM_SIDE_ROLES = [Roles.ASSISTANT, Roles.TOOL_CALL]


class InvokeCost(BaseModel):
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    audio_prompt_tokens: int = 0
    audio_completion_tokens: int = 0
    
    # Rates (USD per token, recorded at invocation time)
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cache_read_input_token_cost: float = 0.0
    
    # Calculated Dollar Costs
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    cost: float = 0.0  # Total combined cost

    def __str__(self):
        rates = []
        if self.input_cost_per_token: rates.append(f"In: ${self.input_cost_per_token:.7f}/tok")
        if self.output_cost_per_token: rates.append(f"Out: ${self.output_cost_per_token:.7f}/tok")
        rates_str = f" | Rates: [{', '.join(rates)}]" if rates else ""
        
        return (f"Tokens: {self.total_tokens} (Prompt: {self.prompt_tokens} "
                f"[Cached: {self.cached_prompt_tokens}], "
                f"Completion: {self.completion_tokens} "
                f"[Reasoning: {self.reasoning_tokens}]){rates_str}\n"
                f"Cost Breakdown: Prompt: ${self.prompt_cost:.6f}, Completion: ${self.completion_cost:.6f} | "
                f"Total Cost: ${self.cost:.6f}")


