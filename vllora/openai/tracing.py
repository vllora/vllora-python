from pdb import run
from agents import RunConfig, RunHooks, TContext
from agents.tracing.span_data import SpanData
from ..core.tracing import vLLoraTracing
from ..core.events import send_vllora_event_sync
from typing import Any, Optional
import os

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.instrumentation.openai_agents._processor import OpenInferenceTracingProcessor

from agents.tracing.setup import GLOBAL_TRACE_PROVIDER
from agents.tracing.processors import BackendSpanExporter
from agents.tracing import Span, Trace
from agents.tracing.create import agent_span as original_agent_span
from agents.tracing.span_data import AgentSpanData

from openai import AsyncOpenAI
from agents.run import DEFAULT_MAX_TURNS, Runner, TraceCtxManager

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry import trace
from opentelemetry.trace.propagation import set_span_in_context


import uuid

original_post = AsyncOpenAI.post
original_init = AsyncOpenAI.__init__

original_on_span_start = OpenInferenceTracingProcessor.on_span_start
original_on_trace_start = OpenInferenceTracingProcessor.on_trace_start

original_runner_run = Runner.run

class RunSpanData(SpanData):
    __slots__ = "name"

    def __init__(
        self,
        name: str,
    ):
        self.name = name

    @property
    def type(self) -> str:
        return "run"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
        }

def async_openai_init(self, *args, **kwargs):
    """
    Monkey-patched __init__ that uses VLLORA_API_KEY instead of OPENAI_API_KEY
    and VLLORA_API_BASE_URL instead of OPENAI_BASE_URL if not set in environment variables.
    """
    # If api_key is not explicitly provided and OPENAI_API_KEY is not set,
    # use VLLORA_API_KEY as fallback, or "no_key" if none available
    if 'api_key' not in kwargs or kwargs['api_key'] is None:
        if not os.environ.get("OPENAI_API_KEY"):
            vllora_api_key = os.environ.get("VLLORA_API_KEY")
            if vllora_api_key:
                kwargs['api_key'] = vllora_api_key
            else:
                kwargs['api_key'] = "no_key"
    
    # If base_url is not explicitly provided and OPENAI_BASE_URL is not set,
    # use VLLORA_API_BASE_URL as fallback
    if 'base_url' not in kwargs or kwargs['base_url'] is None:
        if not os.environ.get("OPENAI_BASE_URL"):
            vllora_base_url = os.environ.get("VLLORA_API_BASE_URL")
            if vllora_base_url:
                kwargs['base_url'] = vllora_base_url
    
    # Call the original __init__
    original_init(self, *args, **kwargs)


def post(self, *args, **kwargs):
    span = trace.get_current_span()

    ctx = set_span_in_context(span)

    headers = kwargs.get('options', {}).get('headers', {})
    TraceContextTextMapPropagator().inject(headers, ctx)

    run_id = span._attributes.get("vllora.run_id")
    thread_id = span._attributes.get("vllora.thread_id")

    headers["x-run-id"] = run_id
    headers["x-thread-id"] = thread_id
    
    kwargs['options']['headers'] = headers

    return original_post(self, *args, **kwargs)

def on_span_start(self, span: Span[any]):
    original_on_span_start(self, span)

    if not span.started_at:
        return
    
    trace = GLOBAL_TRACE_PROVIDER.get_current_trace()

    group_id = trace.export()['group_id']
    if not group_id:
        group_id = str(uuid.UUID(trace.trace_id.replace("trace_", "")))

    self._otel_spans[span.span_id].set_attribute("vllora.thread_id", group_id)
    self._otel_spans[span.span_id].set_attribute("vllora.run_id", group_id)

    if self._otel_spans[span.span_id].attributes.get("openinference.span.kind") == "AGENT":
        self._otel_spans[span.span_id].set_attribute("vllora.agent_name", self._otel_spans[span.span_id].name)
        send_vllora_event_sync(self._otel_spans[span.span_id], "agent")

    if self._otel_spans[span.span_id].attributes.get("openinference.span.kind") == "LLM":
        self._otel_spans[span.span_id].set_attribute("vllora.task_name", self._otel_spans[span.span_id].name)
        send_vllora_event_sync(self._otel_spans[span.span_id], "task")

    if self._otel_spans[span.span_id].attributes.get("openinference.span.kind") == "TOOL":
        self._otel_spans[span.span_id].set_attribute("vllora.tool_name", self._otel_spans[span.span_id].name)
        send_vllora_event_sync(self._otel_spans[span.span_id], "tool")

def on_trace_start(self, trace: Trace):
    otel_span = self._tracer.start_span(
            name="run",
        )

    group_id = trace.export()['group_id']
    if not group_id:
        group_id = str(uuid.UUID(trace.trace_id.replace("trace_", "")))

    otel_span.set_attribute("vllora.thread_id", group_id)
    otel_span.set_attribute("vllora.run_id", group_id)

    self._root_spans[trace.trace_id] = otel_span

    send_vllora_event_sync(otel_span, "run")

def init(collector_endpoint: Optional[str] = None, api_key: Optional[str] = None, project_id: Optional[str] = None):
    import agents.tracing.create
    
    tracer = vLLoraTracing(collector_endpoint, api_key, project_id, "openai")
    
    processor = tracer.get_processor()

    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(processor)

    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

    OpenInferenceTracingProcessor.on_span_start = on_span_start
    OpenInferenceTracingProcessor.on_trace_start = on_trace_start
    
    # Monkey patch AsyncOpenAI to use VLLORA_API_KEY instead of OPENAI_API_KEY
    AsyncOpenAI.__init__ = async_openai_init
    
    # Inject trace headers
    AsyncOpenAI.post = post
    
    # Disable span export
    BackendSpanExporter.export = lambda self, items: None