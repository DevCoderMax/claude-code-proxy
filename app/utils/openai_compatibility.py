"""OpenAI-specific request processing and compatibility fixes."""
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def process_openai_request(litellm_request: Dict[str, Any]):
    """Process and fix OpenAI-specific request requirements."""
    logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
    
    # Process each message for OpenAI compatibility
    for i, msg in enumerate(litellm_request["messages"]):
        # Handle complex content blocks
        if "content" in msg and isinstance(msg["content"], list):
            # Check for messages with only tool_result content
            if _is_only_tool_result_message(msg["content"]):
                litellm_request["messages"][i]["content"] = _extract_tool_result_text(msg["content"])
                continue
        
        # Convert content blocks to simple strings
        if "content" in msg and isinstance(msg["content"], list):
            litellm_request["messages"][i]["content"] = _convert_content_blocks_to_text(msg["content"])
        
        # Handle None content
        elif msg.get("content") is None:
            litellm_request["messages"][i]["content"] = "..."
        
        # Remove unsupported fields
        _remove_unsupported_fields(msg)
    
    # Final validation pass
    _validate_message_content(litellm_request["messages"])

def _is_only_tool_result_message(content: list) -> bool:
    """Check if message contains only tool_result blocks."""
    if not content:
        return False
    
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            return False
    
    return True

def _extract_tool_result_text(content: list) -> str:
    """Extract text from tool_result blocks."""
    all_text = ""
    
    for block in content:
        all_text += "Tool Result:\n"
        result_content = block.get("content", [])
        
        if isinstance(result_content, list):
            for item in result_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    all_text += item.get("text", "") + "\n"
                elif isinstance(item, dict):
                    try:
                        item_text = item.get("text", json.dumps(item))
                        all_text += item_text + "\n"
                    except:
                        all_text += str(item) + "\n"
        elif isinstance(result_content, str):
            all_text += result_content + "\n"
        else:
            try:
                all_text += json.dumps(result_content) + "\n"
            except:
                all_text += str(result_content) + "\n"
    
    return all_text.strip() or "..."

def _convert_content_blocks_to_text(content: list) -> str:
    """Convert complex content blocks to simple string."""
    text_content = ""
    
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            
            if block_type == "text":
                text_content += block.get("text", "") + "\n"
            
            elif block_type == "tool_result":
                tool_id = block.get("tool_use_id", "unknown")
                text_content += f"[Tool Result ID: {tool_id}]\n"
                text_content += _extract_nested_tool_result_content(block.get("content", []))
            
            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_id = block.get("id", "unknown")
                tool_input = json.dumps(block.get("input", {}))
                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
            
            elif block_type == "image":
                text_content += "[Image content - not displayed in text format]\n"
    
    return text_content.strip() or "..."

def _extract_nested_tool_result_content(result_content) -> str:
    """Extract text from nested tool result content."""
    text = ""
    
    if isinstance(result_content, list):
        for item in result_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "") + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    text += item.get("text", "") + "\n"
                else:
                    try:
                        text += json.dumps(item) + "\n"
                    except:
                        text += str(item) + "\n"
    elif isinstance(result_content, dict):
        if result_content.get("type") == "text":
            text += result_content.get("text", "") + "\n"
        else:
            try:
                text += json.dumps(result_content) + "\n"
            except:
                text += str(result_content) + "\n"
    elif isinstance(result_content, str):
        text += result_content + "\n"
    else:
        try:
            text += json.dumps(result_content) + "\n"
        except:
            text += str(result_content) + "\n"
    
    return text

def _remove_unsupported_fields(msg: Dict[str, Any]):
    """Remove fields that OpenAI doesn't support in messages."""
    supported_fields = ["role", "content", "name", "tool_call_id", "tool_calls"]
    
    for key in list(msg.keys()):
        if key not in supported_fields:
            logger.warning(f"Removing unsupported field from message: {key}")
            del msg[key]

def _validate_message_content(messages: list):
    """Final validation of message content."""
    for i, msg in enumerate(messages):
        logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
        
        # Handle remaining list content
        if isinstance(msg.get("content"), list):
            logger.warning(f"CRITICAL: Message {i} still has list content after processing")
            messages[i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
        
        # Handle None content
        elif msg.get("content") is None:
            logger.warning(f"Message {i} has None content - replacing with placeholder")
            messages[i]["content"] = "..."