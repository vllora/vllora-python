from typing import Dict, Any
import os
import asyncio
import httpx


async def send_vllora_event(span, operation: str, attributes: Dict[str, Any] = None):
    """Send span event to vLLora events API (non-blocking)"""
    api_base_url = os.getenv("VLLORA_API_BASE_URL")
    if not api_base_url:
        return
    
    try:
        span_context = span.get_span_context()
        span_id = format(span_context.span_id, '016x')
        trace_id_hex = format(span_context.trace_id, '032x')
        
        parent_span_id = None
        if hasattr(span, 'parent') and span.parent:
            parent_span_id = format(span.parent.span_id, '016x')
        
        event_attributes = attributes or {}
        if span.attributes:
            event_attributes.update(dict(span.attributes))
        
        event_data = {
            "span_id": span_id,
            "trace_id": trace_id_hex,
            "parent_span_id": parent_span_id,
            "operation": operation,
            "attributes": event_attributes
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        api_key = os.getenv("VLLORA_API_KEY")
        project_id = os.getenv("VLLORA_PROJECT_ID")
        
        if api_key:
            headers["x-api-key"] = api_key
        if project_id:
            headers["x-project-id"] = project_id
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_base_url.replace('/v1', '')}/events",
                json=event_data,
                headers=headers,
                timeout=5
            )
        
        if response.status_code != 200:
            print(f"Error sending event to API: {response.status_code}")
            print(f"Event data: {event_data}")
            print(f"Event attributes: {event_attributes}")
            print(f"Event headers: {headers}")
            print(f"Event response: {response.text}")
            raise Exception(f"Error sending event to API: {response.status_code}")
    except Exception as e:
        print(f"Error sending event to API: {e}")
        raise e


def send_vllora_event_sync(span, operation: str, attributes: Dict[str, Any] = None):
    """Send span event to vLLora events API (synchronous wrapper for backward compatibility)"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a task
            asyncio.create_task(send_vllora_event(span, operation, attributes))
        else:
            # If no event loop is running, run the async function
            loop.run_until_complete(send_vllora_event(span, operation, attributes))
    except RuntimeError:
        # No event loop exists, create a new one
        asyncio.run(send_vllora_event(span, operation, attributes))




