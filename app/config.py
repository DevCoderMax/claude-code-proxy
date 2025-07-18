"""Configuration management for the Anthropic proxy server."""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration."""
    
    # API Keys
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
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