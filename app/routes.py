"""API route handlers."""
import json
import time
import logging
import traceback
from typing import Any, Dict, Union

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, Response
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
    """Handle message creation requests by preparing, executing, and processing the request."""
    try:
        litellm_request, display_model = await _prepare_litellm_request(request, raw_request)
        litellm_response = await _execute_litellm_completion(litellm_request, request.stream)
        return _process_litellm_response(litellm_response, request, litellm_request)
    except Exception as e:
        return _handle_error(e)

async def _prepare_litellm_request(request: MessagesRequest, raw_request: Request) -> tuple[dict, str]:
    """Prepare the LiteLLM request from the original Anthropic request."""
    body = await raw_request.body()
    original_model = json.loads(body.decode('utf-8')).get("model", "unknown")
    display_model = original_model.split("/")[-1]

    logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

    litellm_request = convert_anthropic_to_litellm(request)
    _set_api_key(litellm_request, request.model)

    if "openai" in litellm_request["model"]:
        process_openai_request(litellm_request)

    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        litellm_request.get('model'),
        len(litellm_request['messages']),
        len(request.tools or []),
        200
    )
    return litellm_request, display_model

async def _execute_litellm_completion(litellm_request: dict, stream: bool) -> Any:
    """Execute the actual call to LiteLLM."""
    if stream:
        return await litellm.acompletion(**litellm_request)
    
    start_time = time.time()
    response = litellm.completion(**litellm_request)
    logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
    return response

def _process_litellm_response(
    litellm_response: Any,
    request: MessagesRequest,
    litellm_request: dict
) -> Union[Response, StreamingResponse]:
    """Process the LiteLLM response, converting it back to Anthropic format."""
    if request.stream:
        return StreamingResponse(
            handle_streaming(litellm_response, request),
            media_type="text/event-stream"
        )
    
    return convert_litellm_to_anthropic(litellm_response, request)

async def count_tokens(request: TokenCountRequest, raw_request: Request):
    """Handle token counting requests."""
    try:
        original_model = request.original_model or request.model
        display_model = original_model.split("/")[-1]

        messages_request = MessagesRequest(
            model=request.model,
            messages=request.messages,
            tools=request.tools,
            # Add other necessary fields with default values
            max_tokens=1,
            stream=False,
        )
        converted_request = convert_anthropic_to_litellm(messages_request)

        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            converted_request.get('model'),
            len(converted_request['messages']),
            len(request.tools or []),
            200
        )
        
        token_count = litellm.token_counter(
            model=converted_request["model"],
            messages=converted_request["messages"],
        )
        return TokenCountResponse(input_tokens=token_count)
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Re-raise as HTTPException to be caught by the generic error handler
        raise HTTPException(status_code=500, detail=f"Failed to count tokens: {e}")

async def root():
    """Root endpoint."""
    return {"message": "Anthropic Proxy for LiteLLM"}

def _set_api_key(litellm_request: dict, model: str):
    """Set the appropriate API key based on the model."""
    provider = model.split('/')[0]
    api_key_map = {
        "openai": Config.get_openai_api_key,
        "gemini": Config.get_gemini_api_key,
    }
    
    # Default to Anthropic if no specific provider is matched
    get_key_func = api_key_map.get(provider, Config.get_anthropic_api_key)
    api_key = get_key_func()
    
    if api_key:
        litellm_request["api_key"] = api_key
        logger.debug(f"Using {provider or 'anthropic'} API key for model: {model}")

def _handle_error(e: Exception) -> Response:
    """Handle and format errors consistently."""
    status_code = getattr(e, "status_code", 500)
    
    error_details = {
        "error": {
            "type": type(e).__name__,
            "message": str(e)
        }
    }
    
    logger.error(
        "Error processing request",
        exc_info=True,
        extra={"error_details": error_details}
    )
    
    # Ensure status_code is a valid integer
    if not isinstance(status_code, int) or not 100 <= status_code < 600:
        status_code = 500
        
    raise HTTPException(status_code=status_code, detail=json.dumps(error_details))