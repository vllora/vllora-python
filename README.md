# vLLora — Debug your agents in real time

[![PyPI version](https://badge.fury.io/py/vllora.svg)](https://badge.fury.io/py/vllora)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Debug your Agents in Real Time

This Python package is an extension of [vLLora](https://github.com/vllora/vllora) — a lightweight, real-time debugging platform for AI agents. This package adds extra tracing and observability to your agent frameworks. It captures detailed framework-level metadata — decisions, latency, and cost — and sends it to your vLLora Gateway instance, giving you a complete picture of how your agents operate across LangChain, OpenAI Agents, Google ADK, CrewAI, and more.


---

## ⚡ Quick Start

### Google ADK Tracing

![Google ADK Tracing](https://raw.githubusercontent.com/vllora/vllora-python/main/assets/traces-adk.png)

```bash
pip install vllora[adk]
```

```python
# Import and initialize vLLora tracing
# First initialize vLLora before defining any agents
from vllora.adk import init
init()

import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    if city.lower() != "new york":
        return {"status": "error", "error_message": f"Weather information for '{city}' is not available."}
    return {"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}

def get_current_time(city: str) -> dict:
    if city.lower() != "new york":
        return {"status": "error", "error_message": f"Sorry, I don't have timezone information for {city}."}
    tz = ZoneInfo("America/New_York")
    now = datetime.datetime.now(tz)
    return {"status": "success", "report": f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'}

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=("Agent to answer questions about the time and weather in a city." ),
    instruction=("You are a helpful agent who can answer user questions about the time and weather in a city."),
    tools=[get_weather, get_current_time],
)
```

### OpenAI Agents Tracing

![OpenAI Agents Tracing](https://raw.githubusercontent.com/vllora/vllora-python/main/assets/traces-openai.png)

When you call `vllora.openai.init()`, the OpenAI client is automatically configured to use your vLLora gateway by setting the client's `base_url` to the value of `VLLORA_API_BASE_URL`.

```bash
pip install vllora[openai]
```

```python
# Import and initialize vLLora tracing
from vllora.openai import init
init()

# Import agent components after initializing tracing
from agents import Agent, Runner, set_default_openai_client, RunConfig
from openai import AsyncOpenAI
import os

# Configure OpenAI client
client = AsyncOpenAI(
    api_key="no_key",
    base_url=os.environ.get("VLLORA_API_BASE_URL"),
)

set_default_openai_client(client)

agent = Agent(
    name="Math Tutor",
    model="gpt-4",
    instruction="You are a math tutor who can help students with their math homework.",
)

# Your agent will be automatically traced by vLLora
response = await Runner.run(agent, input="Hello World")
```

> **Note:** Always initialize vLLora **before** importing any framework-specific classes to ensure proper instrumentation.

---

## 🛠️ Supported Frameworks

| Framework | Installation | Import Pattern | Status |
| --- | --- | --- | --- |
| Google ADK | `pip install vllora[adk]` | `from vllora.adk import init` | ✅ **Supported** |
| OpenAI Agents | `pip install vllora[openai]` | `from vllora.openai import init` | ✅ **Supported** |
| LangChain | `pip install vllora[langchain]` | `from vllora.langchain import init` | 🚧 **Coming Soon** |
| CrewAI | `pip install vllora[crewai]` | `from vllora.crewai import init` | 🚧 **Coming Soon** |
| Agno | `pip install vllora[agno]` | `from vllora.agno import init` | 🚧 **Coming Soon** |

## 🔧 How It Works

vLLora uses intelligent monkey patching to instrument your AI frameworks at runtime:

<details>
<summary><b>👉 Click to see technical details for each framework</b></summary>

### Google ADK
- Patches `Agent.__init__` to inject callbacks
- Tracks agent hierarchies and tool usage
- Maintains thread context across invocations
- Enriches spans with agent metadata

### OpenAI Agents
- Intercepts agent execution via OpenInference instrumentation
- Tracks agent runs and LLM calls
- Propagates trace context through agent interactions
- Correlates spans across agent hierarchies

</details>

## 📦 Installation

```bash
# Install core vLLora package
pip install vllora

# For specific framework tracing - install framework extras
pip install vllora[adk]      # Google ADK tracing
pip install vllora[openai]   # OpenAI Agents tracing

# Install all supported frameworks
pip install vllora[all]
```

## 🔑 Configuration

Set your configuration (optional credentials can be passed directly to the `init()` function):

```bash
export VLLORA_API_BASE_URL="http://localhost:9090"
```

## ⚙️ Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLORA_API_BASE_URL` | Your vLLora gateway URL; used by `vllora.openai.init()` to set the OpenAI client's `base_url` | Required |
| `VLLORA_API_KEY` | Your vLLora API key. Optional for OpenAI routing (falls back to `"no_key"`), but required if your gateway enforces auth or when using `vllora.adk.vllora_llm`. | Optional |
| `VLLORA_TRACING` | Enable/disable tracing | `true` |
| `VLLORA_TRACING_EXPORTERS` | Comma-separated list of exporters | `otlp` |


## API Reference

### Initialization Functions

Each framework has a simple `init()` function that handles all necessary setup:

- `vllora.adk.init()`: Patches Google ADK Agent class with vLLora callbacks
- `vllora.openai.init()`: Initializes OpenAI Agents tracing and sets OpenAI client `base_url` from `VLLORA_API_BASE_URL`

All init functions accept optional parameters for custom configuration (collector_endpoint, api_key, project_id)

## 🛟 Troubleshooting

### Common Issues

1. **Missing Configuration**: Ensure `VLLORA_API_BASE_URL` is set to your vLLora instance
2. **Tracing Not Working**: Check that initialization functions are called before creating agents
4. **Framework Conflicts**: Initialize vLLora integration before other instrumentation

### Debug Mode

Enable console output for debugging:
```bash
export VLLORA_TRACING_EXPORTERS="otlp,console"
```

Disable tracing entirely:
```bash
export VLLORA_TRACING="false"
```

## Development

### Setting up the environment

1. Clone the repository
2. Create a `.env` file with your configuration:
```bash
VLLORA_API_BASE_URL="http://localhost:9090"
VLLORA_API_KEY="no_api_key"
```

## Publishing

```bash
poetry build
poetry publish
```

## Requirements

- Python >= 3.10
- Framework-specific dependencies (installed automatically)
- OpenTelemetry libraries (installed automatically)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/vllora/vllora-python/issues)
- **Documentation**: [vLLora Documentation](https://vllora.dev/docs)

---