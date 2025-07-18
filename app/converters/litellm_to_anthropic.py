"""Convert LiteLLM responses to Anthropic format."""
import json
import uuid
import logging
from typing import Union, Dict, Any
from ..models import MessagesResponse, MessagesRequest, Usage

logger = logging.getLogger(__name__)

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Extract response data
        response_data = _extract_response_data(litellm_response)
        
        # Create content list for Anthropic format
        content = _build_content_blocks(response_data, is_claude_model)
        
        # Get usage information
        usage_info = _extract_usage_info(response_data)
        
        # Map finish reason to stop reason
        stop_reason = _map_finish_reason_to_stop_reason(response_data.get("finish_reason", "stop"))
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_data.get("response_id", f"msg_{uuid.uuid4()}"),
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=usage_info["input_tokens"],
                output_tokens=usage_info["output_tokens"]
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Return fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

def _extract_response_data(litellm_response) -> Dict[str, Any]:
    """Extract data from LiteLLM response object or dict."""
    if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
        # Extract data from ModelResponse object directly
        choices = litellm_response.choices
        message = choices[0].message if choices and len(choices) > 0 else None
        return {
            "content_text": message.content if message and hasattr(message, 'content') else "",
            "tool_calls": message.tool_calls if message and hasattr(message, 'tool_calls') else None,
            "finish_reason": choices[0].finish_reason if choices and len(choices) > 0 else "stop",
            "usage_info": litellm_response.usage,
            "response_id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        }
    else:
        # Handle dict responses
        try:
            response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
        except AttributeError:
            try:
                response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
            except AttributeError:
                response_dict = {
                    "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                    "choices": getattr(litellm_response, 'choices', [{}]),
                    "usage": getattr(litellm_response, 'usage', {})
                }
        
        choices = response_dict.get("choices", [{}])
        message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
        
        return {
            "content_text": message.get("content", ""),
            "tool_calls": message.get("tool_calls", None),
            "finish_reason": choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop",
            "usage_info": response_dict.get("usage", {}),
            "response_id": response_dict.get("id", f"msg_{uuid.uuid4()}")
        }

def _build_content_blocks(response_data: Dict[str, Any], is_claude_model: bool) -> list:
    """Build content blocks from response data."""
    content = []
    content_text = response_data.get("content_text")
    tool_calls = response_data.get("tool_calls")
    
    # Add text content block if present
    if content_text is not None and content_text != "":
        content.append({"type": "text", "text": content_text})
    
    # Add tool calls if present
    if tool_calls and is_claude_model:
        content.extend(_process_tool_calls_for_claude(tool_calls))
    elif tool_calls and not is_claude_model:
        # For non-Claude models, convert tool calls to text format
        tool_text = _convert_tool_calls_to_text(tool_calls)
        if content and content[0]["type"] == "text":
            content[0]["text"] += tool_text
        else:
            content.append({"type": "text", "text": tool_text})
    
    return content

def _process_tool_calls_for_claude(tool_calls) -> list:
    """Process tool calls for Claude models (native tool_use blocks)."""
    content_blocks = []
    
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    
    for idx, tool_call in enumerate(tool_calls):
        logger.debug(f"Processing tool call {idx}: {tool_call}")
        
        # Extract function data
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
            name = function.get("name", "")
            arguments = function.get("arguments", "{}")
        else:
            function = getattr(tool_call, "function", None)
            tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
            name = getattr(function, "name", "") if function else ""
            arguments = getattr(function, "arguments", "{}") if function else "{}"
        
        # Convert string arguments to dict if needed
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                arguments = {"raw": arguments}
        
        logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
        
        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": arguments
        })
    
    return content_blocks

def _convert_tool_calls_to_text(tool_calls) -> str:
    """Convert tool calls to text format for non-Claude models."""
    tool_text = "\n\nTool usage:\n"
    
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    
    for idx, tool_call in enumerate(tool_calls):
        # Extract function data
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
            name = function.get("name", "")
            arguments = function.get("arguments", "{}")
        else:
            function = getattr(tool_call, "function", None)
            tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
            name = getattr(function, "name", "") if function else ""
            arguments = getattr(function, "arguments", "{}") if function else "{}"
        
        # Convert arguments to formatted string
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
                arguments_str = json.dumps(args_dict, indent=2)
            except json.JSONDecodeError:
                arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments, indent=2)
        
        tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
    
    return tool_text

def _extract_usage_info(response_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract usage information from response data."""
    usage_info = response_data.get("usage_info", {})
    
    if isinstance(usage_info, dict):
        return {
            "input_tokens": usage_info.get("prompt_tokens", 0),
            "output_tokens": usage_info.get("completion_tokens", 0)
        }
    else:
        return {
            "input_tokens": getattr(usage_info, "prompt_tokens", 0),
            "output_tokens": getattr(usage_info, "completion_tokens", 0)
        }

def _map_finish_reason_to_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use"
    }
    return mapping.get(finish_reason, "end_turn")