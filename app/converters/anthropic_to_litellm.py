"""Convert Anthropic API requests to LiteLLM format."""
import json
import logging
from typing import Dict, Any
from ..models import MessagesRequest
from ..utils.content_parser import clean_gemini_schema

logger = logging.getLogger(__name__)

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                text_content = ""
                
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_content = _extract_tool_result_content(block)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        processed_block = _process_content_block(block)
                        if processed_block:
                            processed_content.append(processed_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens for OpenAI models to their limit of 16384
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for OpenAI/Gemini model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        litellm_request["tools"] = _convert_tools_to_openai_format(anthropic_request.tools, anthropic_request.model)
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        litellm_request["tool_choice"] = _convert_tool_choice_to_openai_format(anthropic_request.tool_choice)
    
    return litellm_request

def _extract_tool_result_content(block) -> str:
    """Extract content from a tool result block."""
    result_content = ""
    if hasattr(block, "content"):
        if isinstance(block.content, str):
            result_content = block.content
        elif isinstance(block.content, list):
            for content_block in block.content:
                if hasattr(content_block, "type") and content_block.type == "text":
                    result_content += content_block.text + "\n"
                elif isinstance(content_block, dict) and content_block.get("type") == "text":
                    result_content += content_block.get("text", "") + "\n"
                elif isinstance(content_block, dict):
                    if "text" in content_block:
                        result_content += content_block.get("text", "") + "\n"
                    else:
                        try:
                            result_content += json.dumps(content_block) + "\n"
                        except:
                            result_content += str(content_block) + "\n"
        elif isinstance(block.content, dict):
            if block.content.get("type") == "text":
                result_content = block.content.get("text", "")
            else:
                try:
                    result_content = json.dumps(block.content)
                except:
                    result_content = str(block.content)
        else:
            try:
                result_content = str(block.content)
            except:
                result_content = "Unparseable content"
    
    return result_content

def _process_content_block(block) -> Dict[str, Any]:
    """Process a single content block."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "image":
        return {"type": "image", "source": block.source}
    elif block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input
        }
    elif block.type == "tool_result":
        processed_content_block = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
        }
        
        if hasattr(block, "content"):
            if isinstance(block.content, str):
                processed_content_block["content"] = [{"type": "text", "text": block.content}]
            elif isinstance(block.content, list):
                processed_content_block["content"] = block.content
            else:
                processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
        else:
            processed_content_block["content"] = [{"type": "text", "text": ""}]
        
        return processed_content_block
    
    return None

def _convert_tools_to_openai_format(tools, model: str) -> list:
    """Convert Anthropic tools to OpenAI format."""
    openai_tools = []
    is_gemini_model = model.startswith("gemini/")

    for tool in tools:
        if hasattr(tool, 'dict'):
            tool_dict = tool.dict()
        else:
            try:
                tool_dict = dict(tool) if not isinstance(tool, dict) else tool
            except (TypeError, ValueError):
                logger.error(f"Could not convert tool to dict: {tool}")
                continue

        input_schema = tool_dict.get("input_schema", {})
        if is_gemini_model:
            logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
            input_schema = clean_gemini_schema(input_schema)

        openai_tool = {
            "type": "function",
            "function": {
                "name": tool_dict["name"],
                "description": tool_dict.get("description", ""),
                "parameters": input_schema
            }
        }
        openai_tools.append(openai_tool)

    return openai_tools

def _convert_tool_choice_to_openai_format(tool_choice) -> Any:
    """Convert Anthropic tool_choice to OpenAI format."""
    if hasattr(tool_choice, 'dict'):
        tool_choice_dict = tool_choice.dict()
    else:
        tool_choice_dict = tool_choice
        
    choice_type = tool_choice_dict.get("type")
    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "any"
    elif choice_type == "tool" and "name" in tool_choice_dict:
        return {
            "type": "function",
            "function": {"name": tool_choice_dict["name"]}
        }
    else:
        return "auto"