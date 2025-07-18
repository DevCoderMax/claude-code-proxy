"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import logging
from .config import Config, ModelLists

logger = logging.getLogger(__name__)

class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool

def validate_and_map_model(model_name: str, info) -> str:
    """Validate and map model names based on configuration."""
    original_model = model_name
    new_model = model_name
    
    logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{Config.PREFERRED_PROVIDER}', BIG='{Config.BIG_MODEL}', SMALL='{Config.SMALL_MODEL}'")
    
    # Remove provider prefixes for easier matching
    clean_model = model_name
    if clean_model.startswith('anthropic/'):
        clean_model = clean_model[10:]
    elif clean_model.startswith('openai/'):
        clean_model = clean_model[7:]
    elif clean_model.startswith('gemini/'):
        clean_model = clean_model[7:]
    
    # Mapping Logic
    mapped = False
    
    # Map Haiku to SMALL_MODEL based on provider preference
    if 'haiku' in clean_model.lower():
        if Config.PREFERRED_PROVIDER == "google" and Config.SMALL_MODEL in ModelLists.GEMINI_MODELS:
            new_model = f"gemini/{Config.SMALL_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{Config.SMALL_MODEL}"
            mapped = True
    
    # Map Sonnet to BIG_MODEL based on provider preference
    elif 'sonnet' in clean_model.lower():
        if Config.PREFERRED_PROVIDER == "google" and Config.BIG_MODEL in ModelLists.GEMINI_MODELS:
            new_model = f"gemini/{Config.BIG_MODEL}"
            mapped = True
        else:
            new_model = f"openai/{Config.BIG_MODEL}"
            mapped = True
    
    # Add prefixes to non-mapped models if they match known lists
    elif not mapped:
        if clean_model in ModelLists.GEMINI_MODELS and not model_name.startswith('gemini/'):
            new_model = f"gemini/{clean_model}"
            mapped = True
        elif clean_model in ModelLists.OPENAI_MODELS and not model_name.startswith('openai/'):
            new_model = f"openai/{clean_model}"
            mapped = True
    
    if mapped:
        logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
    else:
        if not model_name.startswith(('openai/', 'gemini/', 'anthropic/')):
            logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
        new_model = model_name
    
    # Store the original model in the values dictionary
    values = info.data
    if isinstance(values, dict):
        values['original_model'] = original_model
    
    return new_model

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None
    
    @field_validator('model')
    def validate_model_field(cls, v, info):
        return validate_and_map_model(v, info)

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        return validate_and_map_model(v, info)

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage