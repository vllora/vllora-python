from typing import Optional, Dict, Any
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from typing import Dict, Any
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from .vllora_llm import vLLoraLlm
from ..core.events import send_vllora_event_sync
from opentelemetry import trace
from google.genai import types
import os
import re
import uuid

# Model callbacks
def vllora_after_model_cb(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    current_state = callback_context.state
    current_state_dict = current_state.to_dict()
    session_id = callback_context._invocation_context.session.id
    invocation_id = callback_context._invocation_context.invocation_id
    span = trace.get_current_span()
    if 'init_session_id' not in current_state_dict:
        callback_context.state['init_session_id'] = session_id
        span.set_attribute("vllora.thread_id", session_id)
    else:
        span.set_attribute("vllora.thread_id", current_state_dict['init_session_id'])
    

    span_context = span.get_span_context()
        # Convert trace_id from int to UUID
    run_id = uuid.UUID(int=span_context.trace_id)
    span.set_attribute("vllora.run_id", run_id.__str__())
    return None

def vllora_before_model_cb(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # Read/Write state example
    session_id = callback_context._invocation_context.session.id
    agent_name = callback_context.agent_name    
    invocation_id = callback_context._invocation_context.invocation_id
    
    current_state = callback_context.state
    current_state_dict = current_state.to_dict()
    init_session_id = session_id
    span = trace.get_current_span()

    span_context = span.get_span_context()
        # Convert trace_id from int to UUID
    run_id = uuid.UUID(int=span_context.trace_id)
    
    if 'init_session_id' not in current_state_dict:
        callback_context.state['init_session_id'] = session_id
        span.set_attribute("vllora.thread_id", session_id)
    else:
        init_session_id = current_state_dict['init_session_id']
        span.set_attribute("vllora.thread_id", init_session_id)
   
    span.set_attribute("vllora.run_id", run_id.__str__())

    sequence_invocation_ids : list[str] = []
    # check if current_state_dict have sequence_invocation_ids
    if 'sequence_invocation_ids' in current_state_dict:
        sequence_invocation_ids = current_state_dict['sequence_invocation_ids']
       
    # add invocation_id to sequence_invocation_ids
    sequence_invocation_ids.append(invocation_id)
        
    # update current_state
    callback_context.state['sequence_invocation_ids'] = sequence_invocation_ids 
    # Create a new config dict if needed
    if not hasattr(llm_request, '_additional_args'):
        llm_request._additional_args = {}
    
    # Add session_id and agent_name to the additional args
    if init_session_id is not None and init_session_id != '':
        llm_request._additional_args['session_id'] = init_session_id
    if agent_name is not None and agent_name != '':
        llm_request._additional_args['agent_name'] = agent_name
        
    llm_request._additional_args['invocation_id'] = invocation_id
    
    
    return None # Allow model call to proceed

# Agent callbacks
def vllora_before_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
   session_id = callback_context._invocation_context.session.id
   invocation_id = callback_context._invocation_context.invocation_id
   current_state = callback_context.state
   current_state_dict = current_state.to_dict()
   
   init_session_id = session_id
   span = trace.get_current_span()
   thread_id = None
   if 'init_session_id' not in current_state_dict:
       callback_context.state['init_session_id'] = session_id
       thread_id = session_id
   else:
       thread_id = current_state_dict['init_session_id']
       init_session_id = thread_id
   
   if thread_id is not None:
       span.set_attribute("vllora.thread_id", thread_id)
   
   span_context = span.get_span_context()
    # Convert trace_id from int to UUID
   run_id = uuid.UUID(int=span_context.trace_id)

   # Send event for agent_run operations
   if span.name.startswith("agent_run"):
       match = re.match(r"agent_run\s*\[(.*?)\]", span.name)
       agent_name = match.group(1) if match else ""
       send_vllora_event_sync(span, "agent", {"vllora.agent_name": agent_name, "vllora.thread_id": thread_id, "vllora.run_id": run_id.__str__()})

   sequence_invocation_ids : list[str] = []
   # check if current_state_dict have sequence_invocation_ids
   if 'sequence_invocation_ids' in current_state_dict:
       sequence_invocation_ids = current_state_dict['sequence_invocation_ids']


   # add invocation_id to sequence_invocation_ids
   sequence_invocation_ids.append(invocation_id)
        
   # update current_state
   callback_context.state['sequence_invocation_ids'] = sequence_invocation_ids    
   
   return None

def vllora_after_agent_cb(callback_context: CallbackContext) -> Optional[types.Content]:
    session_id = callback_context._invocation_context.session.id
    current_state = callback_context.state
    current_state_dict = current_state.to_dict()
    span = trace.get_current_span()
    if 'init_session_id' not in current_state_dict:
        span.set_attribute("vllora.thread_id", session_id)
    else:
        span.set_attribute("vllora.thread_id", current_state_dict['init_session_id'])
    return None

# Tool callbacks

def vllora_before_tool_cb( tool: BaseTool, args: Dict[str, Any], tool_context: CallbackContext) -> Optional[Dict]:
    session_id = tool_context._invocation_context.session.id
    invocation_id = tool_context._invocation_context.invocation_id
    span = trace.get_current_span()
    
    current_state = tool_context.state
    current_state_dict = current_state.to_dict()
    span_context = span.get_span_context()
    # Convert trace_id from int to UUID
    run_id = uuid.UUID(int=span_context.trace_id)
    if 'init_session_id' not in current_state_dict:
       tool_context.state['init_session_id'] = session_id
       span.set_attribute("vllora.thread_id", session_id)
       span.set_attribute("vllora.run_id", run_id.__str__())
    else:
        span.set_attribute("vllora.thread_id", current_state_dict['init_session_id'])
        span.set_attribute("vllora.run_id", run_id.__str__())

    sequence_invocation_ids : list[str] = []
    # check if current_state_dict have sequence_invocation_ids
    if 'sequence_invocation_ids' in current_state_dict:
        sequence_invocation_ids = current_state_dict['sequence_invocation_ids']
        
    # remove invocation_id from sequence_invocation_ids
    sequence_invocation_ids.remove(invocation_id)
        
    # update current_state
    tool_context.state['sequence_invocation_ids'] = sequence_invocation_ids    

    return None

def vllora_after_tool_cb(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict) -> Optional[Dict]:
    session_id = tool_context._invocation_context.session.id
    invocation_id = tool_context._invocation_context.invocation_id
    current_state = tool_context.state
    current_state_dict = current_state.to_dict()
    span = trace.get_current_span()
    span_context = span.get_span_context()
    # Convert trace_id from int to UUID
    run_id = uuid.UUID(int=span_context.trace_id)
    if 'init_session_id' not in current_state_dict:
        span.set_attribute("vllora.thread_id", session_id)
        span.set_attribute("vllora.run_id", run_id.__str__())
    else:
        span.set_attribute("vllora.thread_id", current_state_dict['init_session_id'])
        span.set_attribute("vllora.run_id", run_id.__str__())

    return None


original_init = Agent.__init__

# Agent class
def vllora_agent_init(*args, **kwargs):
    # get model from kwargs
    model = kwargs.get('model')
    if model is None:
        raise ValueError("model is required")
    # check if model is string or vlloraLlm
    if isinstance(model, str):
        model = vLLoraLlm(model)
    elif isinstance(model, vLLoraLlm):
        # do nothing
        pass
    else: 
        # raise error current only support string model
        raise ValueError("model must be a string")
        
    kwargs['model'] = model
    
    input_before_agent_callback = kwargs.get('before_agent_callback')
    if input_before_agent_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_before_agent_callback, list):
            # append vllora_before_agent_cb to the list
            input_before_agent_callback.append(vllora_before_agent_cb)
            kwargs['before_agent_callback'] = input_before_agent_callback
        else:
            kwargs['before_agent_callback'] = [input_before_agent_callback, vllora_before_agent_cb]
    else:
        kwargs['before_agent_callback'] = [vllora_before_agent_cb]
        
    input_after_agent_callback = kwargs.get('after_agent_callback')
    if input_after_agent_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_after_agent_callback, list):
            # append vllora_after_agent_cb to the list
            input_after_agent_callback.append(vllora_after_agent_cb)
            kwargs['after_agent_callback'] = input_after_agent_callback
        else:
            kwargs['after_agent_callback'] = [input_after_agent_callback, vllora_after_agent_cb]
    else:
        kwargs['after_agent_callback'] = [vllora_after_agent_cb]
    
    input_before_model_callback = kwargs.get('before_model_callback')
    if input_before_model_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_before_model_callback, list):
            # append vllora_before_model_cb to the list
            input_before_model_callback.append(vllora_before_model_cb)
            kwargs['before_model_callback'] = input_before_model_callback
        else:
            kwargs['before_model_callback'] = [input_before_model_callback, vllora_before_model_cb]
    else:
        kwargs['before_model_callback'] = [vllora_before_model_cb]
        
    input_after_model_callback = kwargs.get('after_model_callback')
    if input_after_model_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_after_model_callback, list):
            # append vllora_after_model_cb to the list
            input_after_model_callback.append(vllora_after_model_cb)
            kwargs['after_model_callback'] = input_after_model_callback
        else:
            kwargs['after_model_callback'] = [input_after_model_callback, vllora_after_model_cb]
    else:
        kwargs['after_model_callback'] = [vllora_after_model_cb]    
    
    
    input_before_tool_callback = kwargs.get('before_tool_callback')
    if input_before_tool_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_before_tool_callback, list):
            # append vllora_before_tool_cb to the list
            input_before_tool_callback.append(vllora_before_tool_cb)
            kwargs['before_tool_callback'] = input_before_tool_callback
        else:
            kwargs['before_tool_callback'] = [input_before_tool_callback, vllora_before_tool_cb]
    else:
        kwargs['before_tool_callback'] = [vllora_before_tool_cb]
    
    input_after_tool_callback = kwargs.get('after_tool_callback')
    if input_after_tool_callback is not None:
        # check if it is a list or a single callback
        if isinstance(input_after_tool_callback, list):
            # append vllora_after_tool_cb to the list
            input_after_tool_callback.append(vllora_after_tool_cb)
            kwargs['after_tool_callback'] = input_after_tool_callback
        else:
            kwargs['after_tool_callback'] = [input_after_tool_callback, vllora_after_tool_cb]
    else:
        kwargs['after_tool_callback'] = [vllora_after_tool_cb]
        
    original_init(*args, **kwargs)

original_run_async = Agent.run_async

# Agent class

def vllora_agent_run_async(*args, **kwargs):
    span = trace.get_current_span()
    # Send event for invocation operations
    if span.name == "invocation":
        span_context = span.get_span_context()
        trace_id = uuid.UUID(int=span_context.trace_id)
        send_vllora_event_sync(span, "run", {"vllora.run_id": trace_id.__str__(), "vllora.thread_id": args[1].session.id})
    
    return original_run_async(*args, **kwargs)

# Store original start_as_current_span method
original_start_as_current_span = None

def vllora_start_as_current_span(self, name, *args, **kwargs):
    """Wrapper for tracer.start_as_current_span to send vLLora events"""
    # Call the original start_as_current_span
    span_context = original_start_as_current_span(self, name, *args, **kwargs)
    
    # Check if this is a call_llm span and send event
    if name and isinstance(name, str) and name.startswith("call_llm"):
        # Get the span from the context manager when it enters
        class SpanEventWrapper:
            def __init__(self, context, span_name):
                self.context = context
                self.span_name = span_name
                
            def __enter__(self):
                # Get parent span BEFORE entering the new span context
                parent_span = trace.get_current_span()
                parent_attributes = {}
                
                if parent_span and hasattr(parent_span, 'get_span_context'):
                    parent_context = parent_span.get_span_context()
                    if parent_context and parent_context.is_valid:
                        if hasattr(parent_span, 'attributes'):
                            parent_attributes = dict(parent_span.attributes) if parent_span.attributes else {}
                
                # Now enter the new span context
                span = self.context.__enter__()
                
                parent_attributes["vllora.task_name"] = span.name
                # Send event for call_llm operations with parent attributes``
                send_vllora_event_sync(span, "task", parent_attributes)
                return span
                
            def __exit__(self, *args):
                return self.context.__exit__(*args)
        
        return SpanEventWrapper(span_context, name)
    
    return span_context

def init_agent():
    global original_start_as_current_span
    
    # Patch the Tracer class from opentelemetry SDK
    from opentelemetry.sdk.trace import Tracer
    
    if hasattr(Tracer, 'start_as_current_span'):
        original_start_as_current_span = Tracer.start_as_current_span
        Tracer.start_as_current_span = vllora_start_as_current_span
    
    # Monkey patch Agent methods
    Agent.__init__ = vllora_agent_init
    Agent.run_async = vllora_agent_run_async