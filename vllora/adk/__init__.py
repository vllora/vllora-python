"""ADK integration module for vLLora."""

def init():
    from .tracing import init
    from .agent import init_agent
    init_agent()
    init()

__all__ = [
    "init"
]