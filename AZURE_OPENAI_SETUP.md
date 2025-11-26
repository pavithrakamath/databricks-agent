# Azure OpenAI Setup Guide

This guide explains how to configure the agent to use Azure OpenAI as the LLM provider.

## Prerequisites

1. **Azure OpenAI Resource**: You need an Azure OpenAI resource created in your Azure subscription
2. **Deployment**: Create a model deployment (e.g., GPT-4, GPT-3.5-turbo) in your Azure OpenAI resource
3. **API Key**: Obtain your Azure OpenAI API key from the Azure portal

## Installation

```bash
pip install langchain-openai
```

## Configuration

### Method 1: Environment Variables (Recommended)

Set the following environment variables:

```bash
# Provider configuration
export LLM_PROVIDER=azure_openai
export LLM_ENDPOINT_NAME=gpt-4-deployment  # Your deployment name

# Azure OpenAI specific configuration
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-deployment  # Optional if LLM_ENDPOINT_NAME is set
export AZURE_OPENAI_API_KEY=your-api-key-here
export AZURE_OPENAI_API_VERSION=2023-05-15  # Optional, defaults to 2023-05-15
```

### Method 2: Configuration File

Edit `config.py`:

```python
# LLM Provider
LLM_PROVIDER = "azure_openai"
LLM_ENDPOINT_NAME = "gpt-4-deployment"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4-deployment"
AZURE_OPENAI_API_KEY = "your-api-key-here"
AZURE_OPENAI_API_VERSION = "2023-05-15"
```

**Note**: For security, prefer environment variables over hardcoding API keys in config files.

### Method 3: Programmatic Configuration

```python
from agent import create_tool_calling_agent
from llm_provider import create_llm
from tools import get_all_tools
from config import SYSTEM_PROMPT
from langgraph.checkpoint.memory import MemorySaver

# Create Azure OpenAI LLM
llm = create_llm(
    provider="azure_openai",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4-deployment",
    api_key="your-api-key",
    api_version="2023-05-15"  # Optional
)

# Get tools
tools = get_all_tools(include_uc_tools=False)

# Create checkpointer
checkpointer = MemorySaver()

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    verify_final_answer=True
)
```

## Finding Your Azure OpenAI Configuration

### Endpoint URL
1. Go to Azure Portal → Your Azure OpenAI resource
2. Navigate to "Keys and Endpoint"
3. Copy the "Endpoint" value (e.g., `https://your-resource.openai.azure.com/`)

### Deployment Name
1. Go to Azure Portal → Your Azure OpenAI resource
2. Navigate to "Deployments"
3. Copy the deployment name (e.g., `gpt-4-deployment`)

### API Key
1. Go to Azure Portal → Your Azure OpenAI resource
2. Navigate to "Keys and Endpoint"
3. Copy either "KEY 1" or "KEY 2"

### API Version
- Default: `2023-05-15`
- Check Azure OpenAI documentation for latest supported version
- Common versions: `2023-05-15`, `2024-02-15-preview`

## Usage Example

```python
from agent import agent

# Agent is configured with Azure OpenAI
config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({
    "messages": [{"role": "user", "content": "List all schemas"}]
}, config)

print(result)
```

## Troubleshooting

### Error: "azure_endpoint is required"
- Ensure `AZURE_OPENAI_ENDPOINT` environment variable is set
- Or pass `azure_endpoint` parameter to `create_llm()`

### Error: "azure_deployment is required"
- Ensure `AZURE_OPENAI_DEPLOYMENT_NAME` or `LLM_ENDPOINT_NAME` is set
- Or pass `azure_deployment` parameter to `create_llm()`

### Error: "api_key is required"
- Ensure `AZURE_OPENAI_API_KEY` environment variable is set
- Or pass `api_key` parameter to `create_llm()`

### Error: "401 Unauthorized"
- Check that your API key is correct
- Verify the API key hasn't expired
- Ensure you're using the correct endpoint URL

### Error: "404 Not Found"
- Verify the deployment name matches exactly
- Check that the deployment exists in your Azure OpenAI resource
- Ensure you're using the correct endpoint URL

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** or secure secret management
3. **Rotate API keys** regularly
4. **Use least privilege** - only grant necessary permissions
5. **Monitor usage** through Azure Portal

## Additional Resources

- [Azure OpenAI Service Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [LangChain Azure OpenAI Integration](https://python.langchain.com/docs/integrations/chat/azure_chat_openai)
- [Azure OpenAI Pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/)

