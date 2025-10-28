import os
from trace import Trace
import uuid

from typing import Optional, Dict
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanProcessor
from opentelemetry.sdk.trace.export import ReadableSpan
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter

# Environment variable constants
ENV_VLLORA_TRACING = "VLLORA_TRACING"
ENV_VLLORA_TRACING_BASE_URL = "VLLORA_TRACING_BASE_URL"
ENV_VLLORA_API_KEY = "VLLORA_API_KEY"
ENV_VLLORA_PROJECT_ID = "VLLORA_PROJECT_ID"
ENV_VLLORA_TRACING_EXPORTERS = "VLLORA_TRACING_EXPORTERS"

# Default values
DEFAULT_COLLECTOR_ENDPOINT = 'http://0.0.0.0:4317'
DEFAULT_EXPORTERS = "otlp"

# Attribute mapping
attribute_to_vllora_attribute_map = {
    "vllora.thread_id": "vllora.thread_id",
    "session.id": "vllora.thread_id"
}

class vLLoraTracing:
    def __init__(self, collector_endpoint: Optional[str] = None, api_key: Optional[str] = None, project_id: Optional[str] = None, client_name: Optional[str] = None, session_id: Optional[str] = None):
        if os.getenv(ENV_VLLORA_TRACING) == "false":
            return

        if collector_endpoint is None:
            collector_endpoint = os.getenv(ENV_VLLORA_TRACING_BASE_URL)
        if collector_endpoint is None:
            collector_endpoint = DEFAULT_COLLECTOR_ENDPOINT

        if api_key is None:
            api_key = os.getenv(ENV_VLLORA_API_KEY)
             
        if project_id is None:
            project_id = os.getenv(ENV_VLLORA_PROJECT_ID)

        self.collector_endpoint = collector_endpoint
        self.api_key = api_key
        self.project_id = project_id
        self.client_name = client_name
        self.session_id = session_id    
    
    def get_processor(self, **kwargs: any):                
        span_exporter = OTLPSpanExporter(endpoint=self.collector_endpoint, headers=[
            ('x-api-key', self.api_key),
            ('x-project-id', self.project_id)
        ])
        span_exporter_console = ConsoleSpanExporter()

        exporters = os.getenv(ENV_VLLORA_TRACING_EXPORTERS, DEFAULT_EXPORTERS).split(",")

        span_exporters = []
        if "otlp" in exporters:
            span_exporters.append(span_exporter)
        if "console" in exporters:
            span_exporters.append(span_exporter_console)
        
        return AttributePropagationSpanProcessor(span_exporters, self.client_name, self.session_id)

class AttributePropagationSpanProcessor(SpanProcessor):
    def __init__(self, span_exporters: list[SpanExporter] = None, client_name: Optional[str] = None, session_id: Optional[str] = None):
        self.span_exporters = span_exporters
        self.span_attributes: Dict[str, Dict[str, str]] = {}
        self.client_name = client_name
        self.session_id = session_id
    
    def on_start(self, span: ReadableSpan, parent_context = None):
        span_context = span.get_span_context()
        trace_id = uuid.UUID(int=span_context.trace_id)

        if trace_id not in self.span_attributes:
            self.span_attributes[trace_id] = {}

        if "vllora.thread_id" not in span.attributes and self.session_id:
            span.set_attribute("vllora.thread_id", self.session_id)

        for adk_attribute, vllora_attribute in attribute_to_vllora_attribute_map.items():
            if vllora_attribute not in self.span_attributes[trace_id]:
                if adk_attribute in span.attributes:
                    self.span_attributes[trace_id][vllora_attribute] = span.attributes[adk_attribute].replace("e-", "", 1) if isinstance(span.attributes[adk_attribute], str) and span.attributes[adk_attribute].startswith("e-") else span.attributes[adk_attribute]
           
    def on_end(self, span: ReadableSpan):
        # Check if the span has the attribute we want to propagate
        span_context = span.get_span_context()
        trace_id = uuid.UUID(int=span_context.trace_id)

        if trace_id not in self.span_attributes:
            self.span_attributes[trace_id] = {}

        for adk_attribute, vllora_attribute in attribute_to_vllora_attribute_map.items():
            if vllora_attribute not in self.span_attributes[trace_id]:
                if adk_attribute in span.attributes:
                    self.span_attributes[trace_id][vllora_attribute] = span.attributes[adk_attribute].replace("e-", "", 1) if isinstance(span.attributes[adk_attribute], str) and span.attributes[adk_attribute].startswith("e-") else span.attributes[adk_attribute]
        
        # Check if we have stored attributes for this trace and apply them
        if trace_id in self.span_attributes:
            for adk_attribute, vllora_attribute in attribute_to_vllora_attribute_map.items():
                if vllora_attribute not in span.attributes:
                    if vllora_attribute in self.span_attributes[trace_id]:
                        span._attributes[vllora_attribute] = self.span_attributes[trace_id][vllora_attribute]

        if self.client_name is not None and self.client_name != '':
            span._attributes["vllora.client_name"] = self.client_name
        else:
            span._attributes["vllora.client_name"] = "unknown"

        if "vllora.run_id" not in span.attributes:
            span._attributes["vllora.run_id"] = str(trace_id)

        if "vllora.thread_id" not in span.attributes and self.session_id:
            span._attributes["vllora.thread_id"] = self.session_id

        if span._name == "invocation":
            span._name = "run"

        if span._name.startswith("agent_run"):
            # Extract agent name from the span name if it follows the pattern "agent_run [name]"
            import re
            match = re.match(r"agent_run\s*\[(.*?)\]", span._name)
            agent_name = match.group(1) if match else span._attributes.get("agent.name", "")

            span._name = "agent"
            span._attributes["vllora.agent_name"] = agent_name

        if span._attributes.get("openinference.span.kind") == "AGENT":
            span._attributes["vllora.agent_name"] = span._name
            span._name = "agent"

        if span._name.startswith("call_llm"):
            span._name = "task"
            span._attributes["vllora.task_name"] = "call_llm"

        if span._attributes.get("openinference.span.kind") == "LLM":
            span._attributes["vllora.task_name"] = span._name
            span._name = "task"

        if span._attributes.get("openinference.span.kind") == "TOOL":
            span._attributes["vllora.tool_name"] = span._name
            span._name = "tool"

        for span_exporter in self.span_exporters:
            span_exporter.export([span])