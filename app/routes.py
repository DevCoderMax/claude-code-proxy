"""API route handlers."""
import json
import time
import logging
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
import litellm

from .models import MessagesRequest, TokenCountRequest, TokenCountResponse
from .converters.anthropic_to_litellm import convert_anthropic_to_litellm
from .converters.litellm_to_anthropic import convert_litellm_to_anthropic
from .streaming import handle_streaming
from .config import Config
from .logging_config import log_request_beautifully
from .utils.openai_compatibility import process_openai_request

logger = logging.getLogger(__name__)

async def create_message(request: MessagesRequest, raw_request: Request):
    """Handle message creation requests."""
    try:
        # Parse raw request body to get original model
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get display name for logging
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Set appropriate API key based on model
        _set_api_key(litellm_request, request.model)
        
        # Process OpenAI-specific requirements
        if "openai" in litellm_request["model"]:
            process_openai_request(litellm_request)
        
        # Log request details
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", 
            raw_request.url.path, 
            display_model, 
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200
        )
        
        # Handle streaming vs regular completion
        if request.stream:
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            return convert_litellm_to_anthropic(litellm_response, request)
                
    except Exception as e:
        return _handle_error(e)

async def count_tokens(request: TokenCountRequest, raw_request: Request):
    """Handle token counting requests."""
    try:
        original_model = request.original_model or request.model
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model
        
        # Convert to MessagesRequest for processing
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Log request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            converted_request.get('model'),
            len(converted_request['messages']),
            num_tools,
            200
        )
        
        # Count tokens using LiteLLM
        try:
            from litellm import token_counter
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            return TokenCountResponse(input_tokens=token_count)
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            return TokenCountResponse(input_tokens=1000)  # Fallback
            
    except Exception as e:
        return _handle_error(e)

async def root():
    """Root endpoint."""
    return {"message": "Anthropic Proxy for LiteLLM"}

def _set_api_key(litellm_request: dict, model: str):
    """Set the appropriate API key based on the model."""
    if model.startswith("openai/"):
        api_key = Config.get_openai_api_key()
        litellm_request["api_key"] = api_key
        logger.debug(f"Using OpenAI API key for model: {model}")
    elif model.startswith("gemini/"):
        api_key = Config.get_gemini_api_key()
        litellm_request["api_key"] = api_key
        logger.debug(f"Using Gemini API key for model: {model}")
    else:
        api_key = Config.get_anthropic_api_key()
        litellm_request["api_key"] = api_key
        logger.debug(f"Using Anthropic API key for model: {model}")

def _handle_error(e: Exception) -> HTTPException:
    """Handle and format errors consistently."""
    import traceback
    
    error_details = {
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc()
    }
    
    # Check for LiteLLM-specific attributes with safe serialization
    for attr in ['message', 'status_code', 'llm_provider', 'model']:
        if hasattr(e, attr):
            value = getattr(e, attr)
            error_details[attr] = str(value) if value is not None else None
    
    # Handle response attribute separately (might not be JSON serializable)
    if hasattr(e, 'response'):
        response = getattr(e, 'response')
        if response is not None:
            error_details['response'] = str(response)
    
    # Check for additional exception details with safe serialization
    if hasattr(e, '__dict__'):
        for key, value in e.__dict__.items():
            if key not in error_details and key not in ['args', '__traceback__', 'response']:
                try:
                    # Try to serialize the value
                    json.dumps(value)
                    error_details[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    error_details[key] = str(value)
    
    # Safe JSON logging
    try:
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
    except (TypeError, ValueError) as json_error:
        logger.error(f"Error processing request (JSON serialization failed): {error_details}")
        logger.error(f"JSON serialization error: {json_error}")
    
    # Format error message
    error_message = f"Error: {str(e)}"
    if 'message' in error_details and error_details['message']:
        error_message += f"\nMessage: {error_details['message']}"
    if 'response' in error_details and error_details['response']:
        error_message += f"\nResponse: {error_details['response']}"
    
    status_code = error_details.get('status_code', 500)
    if not isinstance(status_code, int):
        status_code = 500
        
    raise HTTPException(status_code=status_code, detail=error_message)