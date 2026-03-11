from enum import Enum
from typing import List, Dict, Any, Optional
import datetime as dt
import tiktoken
from tiktoken.model import encoding_name_for_model
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

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
    TOOL = 'tool'
    TOOL_CALL = 'tool_call'

    @property
    def openai(self):
        if self == Roles.SYSTEM:
            return 'developer'
        return self.value

class Providers(str, Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'

class Modalities(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    FUNCTION_CALL = 'function_call'

class Features(str, Enum):
    FUNCTION_CALL = 'function_call'
    STRUCTURED_OUTPUT = 'structured_output'
    STREAMING = 'streaming'
    FINETUNING = 'finetuning'
    DISTILLATION = 'distillation'
    PREDICTED_OUTPUT = 'predicted_output'
    COMPUTER_USE = 'computer_use'
    WEB_SEARCH = 'web_search'

class APITypes(str, Enum):
    COMPLETION = 'completion'
    RESPONSE = 'response'

class Snapshot(BaseModel):
    name: str
    date: str

    @property
    def dt(self):
        return dt.datetime.strptime(self.date, '%Y-%m-%d')

class CompletionCost(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost: float = 0.0

    def __str__(self):
        return (f"Prompt tokens: {self.prompt_tokens}, \n"
                f"Completion tokens: {self.completion_tokens}, \n"
                f"Cached prompt tokens: {self.cached_prompt_tokens}, \n"
                f"Cost: {self.cost:.4f} USD")

class ModelCard(BaseModel):
    name: str
    provider: Providers
    snapshots: List[Snapshot]
    max_tokens: int
    max_output_tokens: int
    input_price: float # per 1M tokens
    cached_input_price: float # per 1M tokens
    output_price: float # per 1M tokens
    knowledge_cutoff: Optional[str] = None
    features: List[Features] = Field(default_factory=list)
    input_modalities: List[Modalities] = Field(default_factory=lambda: [Modalities.TEXT])
    is_reasoning: bool = False
    base_url: Optional[str] = None

    @property
    def snapshot_dict(self):
        return {s.name: s for s in self.snapshots}

    @property
    def latest_snapshot(self):
        return sorted(self.snapshots, key=lambda x: x.dt)[-1]

    def check_args(self, args: Dict[str, Any]):
        if args is None:
            return
        max_tokens = args.get('max_completion_tokens') or args.get('max_output_tokens')
        if max_tokens is not None and max_tokens > self.max_output_tokens:
            raise ValueError(
                f"Requested max tokens ({max_tokens}) exceeds limit for model {self.name} ({self.max_output_tokens})"
            )

    def cost(self, usage: Dict[str, float]) -> CompletionCost:
        prompt_tokens = int(usage.get('prompt_tokens', 0))
        completion_tokens = int(usage.get('completion_tokens', 0))
        cached_prompt_tokens = int(usage.get('cached_prompt_tokens', 0))
        billable_prompt = max(prompt_tokens - cached_prompt_tokens, 0)
        prompt_cost = (billable_prompt / 1_000_000) * self.input_price
        cached_cost = (cached_prompt_tokens / 1_000_000) * self.cached_input_price
        completion_cost = (completion_tokens / 1_000_000) * self.output_price
        return CompletionCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cost=prompt_cost + cached_cost + completion_cost,
        )


MODEL_CARDS: Dict[str, ModelCard] = {}

def register_model_card(card: ModelCard):
    if card.name in MODEL_CARDS:
        logging.warning(f"Model card \"{card.name}\" already registered. Overwriting.")
    MODEL_CARDS[card.name] = card


# user can also provide a custom model cards file, same name model will be overwritten
def load_model_cards_from_file(path: str):
    models_file = Path(path)
    if not models_file.exists():
        logging.warning(f"Model cards file \"{models_file}\" not found")
        return

    with open(models_file, "rb") as f:
        data = tomllib.load(f)
    
    for model_data in data.get("models", []):
        # Convert string enums to Enum objects
        if "provider" in model_data:
            model_data["provider"] = Providers(model_data["provider"])
        
        if "features" in model_data:
            model_data["features"] = [Features(f) for f in model_data["features"]]
            
        if "input_modalities" in model_data:
            model_data["input_modalities"] = [Modalities(m) for m in model_data["input_modalities"]]
            
        # Handle snapshots
        if "snapshots" in model_data:
            model_data["snapshots"] = [Snapshot(**s) for s in model_data["snapshots"]]
            
        card = ModelCard(**model_data)
        register_model_card(card)

# Define standard models
def _load_builtin_model_cards_from_file():
    models_file = Path(__file__).parent / "models.toml"
    load_model_cards_from_file(models_file)

_load_builtin_model_cards_from_file()


LLM_SIDE_ROLES = [Roles.ASSISTANT, Roles.TOOL_CALL]

def find_model_card(name: str) -> ModelCard:
    if name in MODEL_CARDS:
        return MODEL_CARDS[name]
    # Fallback or search logic could go here
    # For now, return a default or raise error
    # If name matches a snapshot, find the parent card
    for card in MODEL_CARDS.values():
        if name in card.snapshot_dict:
            return card
    
    # If not found, maybe create a generic one or raise
    # For robustness, let's return a generic card if not found, or raise
    raise ValueError(f"Model card for '{name}' not found")
