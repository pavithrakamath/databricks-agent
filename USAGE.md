# Agent Usage Guide

This guide explains how to use the agent in both Databricks and standalone environments.

## Configuration

The agent supports configuration via environment variables or a `.env` file. Environment variables are automatically loaded from `.env` if present.

### Using .env File

Create a `.env` file in the project root:

```bash
# .env
LLM_PROVIDER=azure_openai
LLM_ENDPOINT_NAME=gpt-4-deployment
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2023-05-15
AGENT_EXECUTION_MODE=standalone
```

The agent will automatically load these values when imported.

## Environment Detection

The agent automatically detects whether it's running in Databricks or standalone mode. You can also explicitly set the mode using environment variables.

### Auto-Detection

The agent checks for Databricks environment variables:
- `DATABRICKS_RUNTIME_VERSION`
- `DATABRICKS_ROOT_VIRTUALENV_ENV`
- `SPARK_DATABRICKS_WORKSPACE_URL`

If any of these are present, it runs in Databricks mode. Otherwise, it runs in standalone mode.

### Manual Configuration

Set the execution mode via environment variable:

```bash
# Force Databricks mode
export AGENT_EXECUTION_MODE=databricks

# Force standalone mode
export AGENT_EXECUTION_MODE=standalone

# Auto-detect (default)
export AGENT_EXECUTION_MODE=auto
```

## LLM Provider Configuration

The agent supports multiple LLM providers. Configure via environment variables or code.

### Environment Variables

```bash
# Set provider
export LLM_PROVIDER=databricks  # or "openai", "azure_openai", "anthropic", "ollama"

# Set endpoint/model name
export LLM_ENDPOINT_NAME=my-test-endpoint  # For Databricks
export LLM_ENDPOINT_NAME=gpt-4  # For OpenAI
export LLM_ENDPOINT_NAME=gpt-4-deployment  # For Azure OpenAI
export LLM_ENDPOINT_NAME=claude-3-5-sonnet-20241022  # For Anthropic

# Azure OpenAI specific configuration
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-deployment
export AZURE_OPENAI_API_KEY=your-api-key
export AZURE_OPENAI_API_VERSION=2023-05-15  # Optional, defaults to 2023-05-15
```

### Supported Providers

1. **Databricks** (default in Databricks environment)
   - Requires: `databricks-langchain`
   - Uses: Databricks model serving endpoints
   - Example: `LLM_ENDPOINT_NAME=databricks-claude-3-7-sonnet`

2. **OpenAI**
   - Requires: `langchain-openai`
   - Uses: OpenAI API
   - Example: `LLM_ENDPOINT_NAME=gpt-4` or `gpt-3.5-turbo`

3. **Azure OpenAI**
   - Requires: `langchain-openai`
   - Uses: Azure OpenAI Service
   - Required environment variables:
     - `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint URL
     - `AZURE_OPENAI_DEPLOYMENT_NAME` - Deployment name (or use `LLM_ENDPOINT_NAME`)
     - `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
     - `AZURE_OPENAI_API_VERSION` - API version (optional, defaults to "2023-05-15")
   - Example:
     ```bash
     export LLM_PROVIDER=azure_openai
     export LLM_ENDPOINT_NAME=gpt-4-deployment
     export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
     export AZURE_OPENAI_API_KEY=your-api-key
     ```

4. **Anthropic**
   - Requires: `langchain-anthropic`
   - Uses: Anthropic API
   - Example: `LLM_ENDPOINT_NAME=claude-3-5-sonnet-20241022`

5. **Ollama** (for local models)
   - Requires: `langchain-community`
   - Uses: Local Ollama instance
   - Example: `LLM_ENDPOINT_NAME=llama2`

## Usage Examples

### Databricks Mode

```python
# In Databricks notebook or environment
from agent import agent, AGENT

# Agent is automatically configured for Databricks
# Uses ChatDatabricks with your endpoint
# Includes UC tools (system.ai.python_exec)

# Use the agent
config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "List all schemas"}]}, config)
```

### Standalone Mode

```python
# In standalone Python environment
import os

# Configure for standalone mode
os.environ["AGENT_EXECUTION_MODE"] = "standalone"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_ENDPOINT_NAME"] = "gpt-4"

from agent import agent, AGENT

# Agent is configured for standalone
# Uses OpenAI (or your configured provider)
# Excludes UC tools (not available outside Databricks)

# Use the agent
config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "List all schemas"}]}, config)
```

### Azure OpenAI Mode

```python
# In standalone Python environment
import os

# Configure for Azure OpenAI
os.environ["AGENT_EXECUTION_MODE"] = "standalone"
os.environ["LLM_PROVIDER"] = "azure_openai"
os.environ["LLM_ENDPOINT_NAME"] = "gpt-4-deployment"  # Deployment name
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"  # Optional

from agent import agent, AGENT

# Agent is configured for Azure OpenAI
# Uses Azure OpenAI Service

# Use the agent
config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "List all schemas"}]}, config)
```

Or programmatically:

```python
from agent import create_tool_calling_agent
from llm_provider import create_llm
from tools import get_all_tools
from config import SYSTEM_PROMPT

# Create Azure OpenAI LLM
llm = create_llm(
    provider="azure_openai",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4-deployment",
    api_key="your-api-key",
    api_version="2023-05-15"
)

# Get tools
tools = get_all_tools(include_uc_tools=False)

# Create agent
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    verify_final_answer=True
)
```

### Programmatic Configuration

```python
from agent import create_tool_calling_agent
from llm_provider import create_llm
from tools import get_all_tools
from config import SYSTEM_PROMPT

# Create LLM with specific provider
llm = create_llm(
    provider="openai",
    model_name="gpt-4",
    temperature=0.7
)

# Get tools (exclude UC tools for standalone)
tools = get_all_tools(include_uc_tools=False)

# Create agent
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    verify_final_answer=True
)
```

## MLflow Integration

MLflow integration is **optional** and only available when `mlflow` is installed.

### With MLflow (Databricks or standalone)

```python
from agent import AGENT

# AGENT is a LangGraphResponsesAgent wrapper
# Can be used with MLflow model serving
```

### Without MLflow (Standalone)

```python
from agent import agent

# agent is the raw LangGraph agent
# Use directly without MLflow wrapper
result = agent.invoke({"messages": [...]}, config)
```

## Installation Requirements

### For Databricks Mode

```bash
pip install databricks-langchain databricks-agents mlflow-skinny[databricks]
```

### For Standalone Mode

**Minimum requirements:**
```bash
pip install langchain langchain-core langgraph pyspark
```

**With OpenAI:**
```bash
pip install langchain-openai
```

**With Azure OpenAI:**
```bash
pip install langchain-openai
# Then configure via environment variables (see above)
```

**With Anthropic:**
```bash
pip install langchain-anthropic
```

**With MLflow (optional):**
```bash
pip install mlflow-skinny
```

**With dotenv support:**
```bash
pip install python-dotenv
```

## Human-in-the-Loop Usage

The agent includes built-in human-in-the-loop middleware for tool approval and answer verification.

### Tool Approval

When the agent calls a tool that requires approval (e.g., `execute_sql_query`), execution pauses and waits for approval:

```python
from agent import agent
from middleware import check_interrupt_state, approve_tool_calls, get_pending_tool_calls

config = {"configurable": {"thread_id": "1"}}

# Invoke agent - it will pause if tool approval is needed
agent.invoke({
    "messages": [{"role": "user", "content": "Run SELECT * FROM sales LIMIT 10"}]
}, config)

# Check if agent is waiting for approval
interrupt_info = check_interrupt_state(agent, config)
if interrupt_info and interrupt_info.get("interrupted"):
    # Get pending tool calls
    pending = get_pending_tool_calls(agent, config)
    print(f"Pending tool calls: {pending}")
    
    # Approve all pending tool calls
    result = approve_tool_calls(agent, config, decision="approve")
    
    # Or reject
    # result = approve_tool_calls(agent, config, decision="reject")
    
    # Or edit before approving
    # from middleware import edit_tool_call
    # edit_tool_call(agent, config, tool_call_id="call_xxx", new_args={"query": "SELECT * FROM sales LIMIT 5"})
    # result = approve_tool_calls(agent, config, decision="approve")
```

### Final Answer Verification

The agent can also pause before sending final answers:

```python
from middleware import get_final_answer, approve_final_answer

# After agent generates a final answer, check if it's waiting for approval
final_answer = get_final_answer(agent, config)
if final_answer:
    print(f"Final answer: {final_answer}")
    
    # Approve the answer
    approve_final_answer(agent, config, decision="approve")
    
    # Or edit it
    # approve_final_answer(agent, config, decision="edit", edited_answer="Your edited answer here")
    
    # Or reject it
    # approve_final_answer(agent, config, decision="reject")
```

### Configuring Tool Approval

Edit `config.py` to configure which tools require approval:

```python
MIDDLEWARE_CONFIG = {
    "execute_sql_query": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "required": True  # Requires approval
    },
    "greet_me": False,  # No approval needed
    # ... other tools
}
```

## Features by Mode

| Feature | Databricks Mode | Standalone Mode |
|---------|----------------|-----------------|
| Spark Session | DatabricksSession | SparkSession |
| LLM Provider | ChatDatabricks | OpenAI/Azure OpenAI/Anthropic/etc. |
| UC Tools | ✅ Available | ❌ Not available |
| MLflow Integration | ✅ Available | ✅ Optional |
| Human-in-the-Loop | ✅ Available | ✅ Available |
| SQL Tools | ✅ Available | ✅ Available |
| .env File Support | ✅ Available | ✅ Available |

## Troubleshooting

### Provider Not Found

If you get "No LLM provider available":
1. Install the required provider package
2. Set `LLM_PROVIDER` environment variable
3. Or ensure you're in Databricks environment (auto-detects)

### UC Tools Error

If UC tools fail to load:
- In standalone mode, this is expected - UC tools are Databricks-specific
- Set `include_uc_tools=False` when calling `get_all_tools()`

### Spark Session Issues

If Spark session creation fails:
- Ensure PySpark is installed: `pip install pyspark`
- For Databricks, ensure you're in a Databricks environment
- Use `force_standalone=True` in `get_spark_session()` to force SparkSession

## Project Structure

The codebase is organized into logical modules:

- `agent.py` - Main agent orchestration and MLflow integration
- `config.py` - Configuration constants and settings
- `tools.py` - Agent tools (SQL execution, schema listing, etc.)
- `middleware.py` - Human-in-the-loop middleware helpers
- `llm_provider.py` - Multi-provider LLM abstraction
- `environment.py` - Environment detection utilities
- `security.py` - Security and validation utilities
- `utils.py` - Common utility functions
- `term_expansion.py` - Term expansion utilities

## Migration from Databricks-Only Code

If you have existing code that assumes Databricks:

1. **Replace hardcoded ChatDatabricks:**
   ```python
   # Old code
   from databricks_langchain import ChatDatabricks
   llm = ChatDatabricks(endpoint="my-endpoint")
   
   # New code (works in both modes)
   from llm_provider import create_llm
   llm = create_llm(provider="databricks", endpoint="my-endpoint")
   ```

2. **Update tool initialization:**
   ```python
   # Old code
   from databricks_langchain import UCFunctionToolkit
   uc_toolkit = UCFunctionToolkit(function_names=["system.ai.python_exec"])
   
   # New code (auto-detects environment)
   from tools import get_all_tools
   tools = get_all_tools()  # Includes UC tools in Databricks, excludes in standalone
   ```

3. **Use environment variables or .env file:**
   ```python
   # Old code - hardcoded values
   LLM_ENDPOINT_NAME = "my-endpoint"
   
   # New code - use environment variables or .env file
   # Set in .env file or environment:
   # LLM_PROVIDER=databricks
   # LLM_ENDPOINT_NAME=my-endpoint
   from config import LLM_ENDPOINT_NAME, LLM_PROVIDER
   ```

4. **Handle human-in-the-loop interrupts:**
   ```python
   # New - check for interrupts and approve if needed
   from middleware import check_interrupt_state, approve_tool_calls
   
   result = agent.invoke({"messages": [...]}, config)
   
   # Check if waiting for approval
   interrupt_info = check_interrupt_state(agent, config)
   if interrupt_info and interrupt_info.get("interrupted"):
       approve_tool_calls(agent, config, decision="approve")
