"""Handle streaming responses from LiteLLM and convert to Anthropic format."""
import json
import uuid
import logging
from typing import AsyncGenerator
from .models import MessagesRequest

logger = logging.getLogger(__name__)

async def handle_streaming(response_generator, original_request: MessagesRequest) -> AsyncGenerator[str, None]:
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        # Initialize streaming state
        streaming_state = StreamingState()
        
        # Process each chunk
        async for chunk in response_generator:
            try:
                await _process_chunk(chunk, streaming_state)
                
                # Yield any events generated from this chunk
                for event in streaming_state.get_and_clear_events():
                    yield event
                    
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # Finalize streaming if not already done
        if not streaming_state.has_sent_stop_reason:
            for event in streaming_state.finalize():
                yield event
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error events
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

class StreamingState:
    """Manages the state of streaming response processing."""
    
    def __init__(self):
        self.tool_index = None
        self.current_tool_call = None
        self.tool_content = ""
        self.accumulated_text = ""
        self.text_sent = False
        self.text_block_closed = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.has_sent_stop_reason = False
        self.last_tool_index = 0
        self.events = []
    
    def add_event(self, event: str):
        """Add an event to be yielded."""
        self.events.append(event)
    
    def get_and_clear_events(self) -> list:
        """Get all pending events and clear the list."""
        events = self.events.copy()
        self.events.clear()
        return events
    
    def finalize(self) -> list:
        """Generate final events to close the stream."""
        events = []
        
        # Close any open tool call blocks
        if self.tool_index is not None:
            for i in range(1, self.last_tool_index + 1):
                events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n")
        
        # Close text block if not already closed
        if not self.text_block_closed:
            if self.accumulated_text and not self.text_sent:
                events.append(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': self.accumulated_text}})}\n\n")
            events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")
        
        # Send final message_delta with usage
        usage = {"output_tokens": self.output_tokens}
        events.append(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n")
        events.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")
        events.append("data: [DONE]\n\n")
        
        return events

async def _process_chunk(chunk, state: StreamingState):
    """Process a single chunk from the streaming response."""
    # Check for usage data
    if hasattr(chunk, 'usage') and chunk.usage is not None:
        if hasattr(chunk.usage, 'prompt_tokens'):
            state.input_tokens = chunk.usage.prompt_tokens
        if hasattr(chunk.usage, 'completion_tokens'):
            state.output_tokens = chunk.usage.completion_tokens
    
    # Handle text content and tool calls
    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
        choice = chunk.choices[0]
        delta = getattr(choice, 'delta', getattr(choice, 'message', {}))
        finish_reason = getattr(choice, 'finish_reason', None)
        
        # Process text content
        await _process_text_content(delta, state)
        
        # Process tool calls
        await _process_tool_calls(delta, state)
        
        # Process finish_reason
        if finish_reason and not state.has_sent_stop_reason:
            await _process_finish_reason(finish_reason, state)

async def _process_text_content(delta, state: StreamingState):
    """Process text content from delta."""
    delta_content = None
    
    if hasattr(delta, 'content'):
        delta_content = delta.content
    elif isinstance(delta, dict) and 'content' in delta:
        delta_content = delta['content']
    
    if delta_content is not None and delta_content != "":
        state.accumulated_text += delta_content
        
        if state.tool_index is None and not state.text_block_closed:
            state.text_sent = True
            state.add_event(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n")

async def _process_tool_calls(delta, state: StreamingState):
    """Process tool calls from delta."""
    delta_tool_calls = None
    
    if hasattr(delta, 'tool_calls'):
        delta_tool_calls = delta.tool_calls
    elif isinstance(delta, dict) and 'tool_calls' in delta:
        delta_tool_calls = delta['tool_calls']
    
    if delta_tool_calls:
        # Handle first tool call - close text block if needed
        if state.tool_index is None:
            await _handle_first_tool_call(state)
        
        # Process tool calls
        if not isinstance(delta_tool_calls, list):
            delta_tool_calls = [delta_tool_calls]
        
        for tool_call in delta_tool_calls:
            await _process_single_tool_call(tool_call, state)

async def _handle_first_tool_call(state: StreamingState):
    """Handle the first tool call - close text block appropriately."""
    if state.text_sent and not state.text_block_closed:
        state.text_block_closed = True
        state.add_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")
    elif state.accumulated_text and not state.text_sent and not state.text_block_closed:
        state.text_sent = True
        state.add_event(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': state.accumulated_text}})}\n\n")
        state.text_block_closed = True
        state.add_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")
    elif not state.text_block_closed:
        state.text_block_closed = True
        state.add_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")

async def _process_single_tool_call(tool_call, state: StreamingState):
    """Process a single tool call."""
    # Get the index of this tool call
    current_index = None
    if isinstance(tool_call, dict) and 'index' in tool_call:
        current_index = tool_call['index']
    elif hasattr(tool_call, 'index'):
        current_index = tool_call.index
    else:
        current_index = 0
    
    # Check if this is a new tool or continuation
    if state.tool_index is None or current_index != state.tool_index:
        # New tool call
        state.tool_index = current_index
        state.last_tool_index += 1
        anthropic_tool_index = state.last_tool_index
        
        # Extract function info
        if isinstance(tool_call, dict):
            function = tool_call.get('function', {})
            name = function.get('name', '') if isinstance(function, dict) else ""
            tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
        else:
            function = getattr(tool_call, 'function', None)
            name = getattr(function, 'name', '') if function else ''
            tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
        
        # Start new tool_use block
        state.add_event(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n")
        state.current_tool_call = tool_call
        state.tool_content = ""
    
    # Extract and process function arguments
    arguments = None
    if isinstance(tool_call, dict) and 'function' in tool_call:
        function = tool_call.get('function', {})
        arguments = function.get('arguments', '') if isinstance(function, dict) else ''
    elif hasattr(tool_call, 'function'):
        function = getattr(tool_call, 'function', None)
        arguments = getattr(function, 'arguments', '') if function else ''
    
    if arguments:
        # Process arguments
        try:
            if isinstance(arguments, dict):
                args_json = json.dumps(arguments)
            else:
                json.loads(arguments)  # Validate JSON
                args_json = arguments
        except (json.JSONDecodeError, TypeError):
            args_json = arguments
        
        state.tool_content += args_json if isinstance(args_json, str) else ""
        state.add_event(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': state.last_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n")

async def _process_finish_reason(finish_reason: str, state: StreamingState):
    """Process finish reason and send final events."""
    state.has_sent_stop_reason = True
    
    # Close any open tool call blocks
    if state.tool_index is not None:
        for i in range(1, state.last_tool_index + 1):
            state.add_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n")
    
    # Close text block if needed
    if not state.text_block_closed:
        if state.accumulated_text and not state.text_sent:
            state.add_event(f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': state.accumulated_text}})}\n\n")
        state.add_event(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")
    
    # Map finish reason to stop reason
    stop_reason_mapping = {
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "stop": "end_turn"
    }
    stop_reason = stop_reason_mapping.get(finish_reason, "end_turn")
    
    # Send final events
    usage = {"output_tokens": state.output_tokens}
    state.add_event(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n")
    state.add_event(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")
    state.add_event("data: [DONE]\n\n")