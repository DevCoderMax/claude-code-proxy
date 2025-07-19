"""Configuration management for the Anthropic proxy server."""
import os
import random
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys - support multiple keys separated by commas
    _ANTHROPIC_API_KEYS = [key.strip() for key in os.environ.get("ANTHROPIC_API_KEY", "").split(",") if key.strip()]
    _OPENAI_API_KEYS = [key.strip() for key in os.environ.get("OPENAI_API_KEY", "").split(",") if key.strip()]
    _GEMINI_API_KEYS = [key.strip() for key in os.environ.get("GEMINI_API_KEY", "").split(",") if key.strip()]
    
    @classmethod
    def get_anthropic_api_key(cls) -> Optional[str]:
        """Get a random Anthropic API key from available keys."""
        return random.choice(cls._ANTHROPIC_API_KEYS) if cls._ANTHROPIC_API_KEYS else None
    
    @classmethod
    def get_openai_api_key(cls) -> Optional[str]:
        """Get a random OpenAI API key from available keys."""
        return random.choice(cls._OPENAI_API_KEYS) if cls._OPENAI_API_KEYS else None
    
    @classmethod
    def get_gemini_api_key(cls) -> Optional[str]:
        """Get a random Gemini API key from available keys."""
        return random.choice(cls._GEMINI_API_KEYS) if cls._GEMINI_API_KEYS else None
    
    # Backward compatibility - these will return the first key or None
    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        return self._ANTHROPIC_API_KEYS[0] if self._ANTHROPIC_API_KEYS else None
    
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return self._OPENAI_API_KEYS[0] if self._OPENAI_API_KEYS else None
    
    @property
    def GEMINI_API_KEY(self) -> Optional[str]:
        return self._GEMINI_API_KEYS[0] if self._GEMINI_API_KEYS else None
    
    # Provider preferences
    PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
    
    # Model mappings
    BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
    SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8082
    LOG_LEVEL = "error"

class ModelLists:
    """Model definitions for different providers."""
    
    OPENAI_MODELS: List[str] = [
        "o3-mini",
        "o1",
        "o1-mini", 
        "o1-pro",
        "gpt-4.5-preview",
        "gpt-4o",
        "gpt-4o-audio-preview",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-audio-preview",
        "gpt-4.1",
        "gpt-4.1-mini"
    ]
    
    GEMINI_MODELS: List[str] = [
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash"
    ]